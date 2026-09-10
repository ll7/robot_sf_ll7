"""Fail-closed, outcome-free packet for the #1496 imitation comparison.

The comparison packet is deliberately separate from the training runners.  It freezes the
two arms, matched budgets, split identities, checkpoint/evaluation rules, and the estimand
boundary before a private dataset or a compute allocation is available.  A small synthetic
row inventory can be checked locally, while the nominal private artifact remains an explicit
input to a later worker lane.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from robot_sf.training.oracle_imitation_launch_packet import (
    LaunchPacketError,
    validate_launch_packet,
)
from robot_sf.training.oracle_trace_uri_registry import (
    OracleTraceUriRegistryError,
    validate_trace_uri_registry,
)

COMPARISON_SCHEMA = "oracle-imitation-comparison.v1"
CONSUMER_SCHEMA = "oracle-imitation-dataset-consumer.v1"
CANARY_SCHEMA = "oracle-imitation-startup-canary.v1"
SPLITS = ("train", "validation", "evaluation")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$")
_PRIVATE_URI_PREFIX = "private-artifact://"
_ALLOWED_INVENTORY_STATES = {"complete", "private_required", "unavailable"}
_VALID_ROW_STATUS = "valid"


class ComparisonPacketError(ValueError):
    """Raised when a comparison or dataset-consumer contract is malformed."""


class ComparisonArtifactBlockedError(ComparisonPacketError):
    """Raised when a well-formed contract cannot access a required artifact."""


def _resolve_path(value: str | Path, root: Path) -> Path:
    """Resolve a repository-relative path without changing the filesystem.

    Returns:
        The absolute path represented by ``value``.
    """
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def _load_mapping(path: Path) -> dict[str, Any]:
    """Load one YAML mapping and give callers a stable contract error.

    Returns:
        The parsed YAML mapping.
    """
    if not path.is_file():
        raise ComparisonPacketError(f"manifest is not a regular file: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ComparisonPacketError(f"failed to load YAML manifest: {path.name}") from exc
    if not isinstance(payload, dict):
        raise ComparisonPacketError(f"manifest must be a YAML mapping: {path.name}")
    return payload


def _require_string(payload: dict[str, Any], key: str, *, context: str) -> str:
    """Return a required non-empty string field."""
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ComparisonPacketError(f"{context}.{key} must be a non-empty string")
    return value.strip()


def _require_sha(value: Any, *, field: str) -> str:
    """Return a normalized full SHA-256 digest."""
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value.strip()) is None:
        raise ComparisonPacketError(f"{field} must be a 64-character SHA-256 digest")
    return value.strip().lower()


def _require_id(value: Any, *, field: str) -> str:
    """Return a stable packet or sample identifier."""
    if not isinstance(value, str) or _ID_RE.fullmatch(value.strip()) is None:
        raise ComparisonPacketError(f"{field} must be a lowercase stable identifier")
    return value.strip()


def _integer_list(value: Any, *, field: str) -> list[int]:
    """Parse a non-empty list of distinct integer seeds.

    Returns:
        The normalized integer list.
    """
    if not isinstance(value, list) or not value:
        raise ComparisonPacketError(f"{field} must be a non-empty list")
    result: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise ComparisonPacketError(f"{field}[{index}] must be an integer")
        result.append(int(item))
    if len(set(result)) != len(result):
        raise ComparisonPacketError(f"{field} must not contain duplicate values")
    return result


def _canonical_json(value: Any) -> str:
    """Render a JSON-compatible value deterministically.

    Returns:
        Canonical JSON text with stable key and separator ordering.
    """
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _semantic_digest(payload: dict[str, Any]) -> str:
    """Hash a consumer manifest without trusting a self-reported digest field.

    Returns:
        The SHA-256 digest of the canonical manifest content.
    """
    digest_input = {key: value for key, value in payload.items() if key != "manifest_digest"}
    return hashlib.sha256(_canonical_json(digest_input).encode("utf-8")).hexdigest()


def _safe_relative_path(raw: Any, *, field: str) -> Path:
    """Validate a manifest path before resolving it under an artifact root.

    Returns:
        A validated relative path.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise ComparisonPacketError(f"{field} must be a non-empty repository-relative path")
    path = Path(raw.strip())
    if path.is_absolute() or ".." in path.parts:
        raise ComparisonPacketError(f"{field} must not be absolute or escape its root")
    return path


def _resolve_private_uri(uri: str, root: Path, *, field: str) -> Path:
    """Resolve a private-artifact URI beneath an explicitly supplied root.

    Returns:
        The resolved local artifact path.
    """
    relative = uri.removeprefix(_PRIVATE_URI_PREFIX)
    if (
        not relative
        or relative.startswith("/")
        or "\\" in relative
        or any(part in {"", ".", ".."} for part in relative.split("/"))
    ):
        raise ComparisonPacketError(f"{field} has an invalid private-artifact URI")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ComparisonPacketError(f"{field} escapes the configured artifact root") from exc
    return candidate


def _finite_vector(value: Any, *, field: str) -> list[float]:
    """Validate one flat finite numeric feature or action vector.

    Returns:
        The normalized vector as finite floating-point values.
    """
    if not isinstance(value, list) or not value:
        raise ComparisonPacketError(f"{field} must be a non-empty numeric list")
    result: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ComparisonPacketError(f"{field}[{index}] must be numeric")
        number = float(item)
        if not math.isfinite(number):
            raise ComparisonPacketError(f"{field}[{index}] must be finite")
        result.append(number)
    return result


def _sha256_file(path: Path, *, chunk_size: int = 65_536) -> str:
    """Hash a local artifact incrementally so strict checks do not load it into memory.

    Returns:
        The hexadecimal SHA-256 digest of the file contents.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _check_file(
    path: Path,
    *,
    expected_sha: str,
    expected_size: int | None,
    field: str,
) -> dict[str, Any]:
    """Verify one local regular artifact and return public-safe evidence.

    Returns:
        Public-safe size and digest evidence for the verified artifact.
    """
    if path.is_symlink():
        raise ComparisonPacketError(f"{field} must not be a symlink")
    if not path.is_file():
        raise ComparisonArtifactBlockedError(f"{field} is missing: {path.name}")
    actual_size = path.stat().st_size
    if expected_size is not None and actual_size != expected_size:
        raise ComparisonPacketError(
            f"{field} size mismatch: expected {expected_size}, got {actual_size}"
        )
    digest = _sha256_file(path)
    if digest != expected_sha:
        raise ComparisonPacketError(
            f"{field} digest mismatch: expected {expected_sha}, got {digest}"
        )
    return {"status": "verified", "size_bytes": actual_size, "sha256": digest}


def _validate_contract_dimensions(
    payload: dict[str, Any],
    *,
    field: str,
) -> int | None:
    """Validate an optional vector dimension, preserving unknown private dimensions.

    Returns:
        The positive dimension or ``None`` when the private dimension is undeclared.
    """
    dimension = payload.get("dimension")
    if dimension is None:
        return None
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise ComparisonPacketError(f"{field}.dimension must be a positive integer or null")
    return dimension


def _validate_inventory_row(
    raw_row: Any,
    *,
    index: int,
    split_ids: dict[str, set[str]],
    feature_dimension: int | None,
    action_dimension: int | None,
) -> dict[str, Any]:
    """Validate one complete-inventory row.

    Returns:
        A normalized row with only the fields needed by the canary.
    """
    if not isinstance(raw_row, dict):
        raise ComparisonPacketError(f"sample_inventory.rows[{index}] must be a mapping")
    sample_id = _require_id(raw_row.get("sample_id"), field=f"rows[{index}].sample_id")
    split = raw_row.get("split")
    if split not in SPLITS:
        raise ComparisonPacketError(f"rows[{index}].split must be one of {SPLITS}")
    if sample_id not in split_ids[split]:
        raise ComparisonPacketError(f"row {sample_id} is not declared in its {split} split")
    row_status = raw_row.get("status")
    if row_status != _VALID_ROW_STATUS:
        raise ComparisonPacketError(
            f"row {sample_id} has non-admissible status {row_status!r}; "
            "fallback/degraded/invalid rows fail closed"
        )
    features = _finite_vector(raw_row.get("features"), field=f"row {sample_id}.features")
    action = _finite_vector(raw_row.get("action"), field=f"row {sample_id}.action")
    if feature_dimension is not None and len(features) != feature_dimension:
        raise ComparisonPacketError(
            f"row {sample_id} feature dimension mismatch: expected "
            f"{feature_dimension}, got {len(features)}"
        )
    if action_dimension is not None and len(action) != action_dimension:
        raise ComparisonPacketError(
            f"row {sample_id} action dimension mismatch: expected "
            f"{action_dimension}, got {len(action)}"
        )
    return {"sample_id": sample_id, "split": split, "features": features, "action": action}


def _validate_complete_inventory(
    rows: list[Any],
    *,
    split_ids: dict[str, set[str]],
    feature_dimension: int | None,
    action_dimension: int | None,
) -> dict[str, Any]:
    """Validate all rows in a complete inventory.

    Returns:
        A normalized complete-inventory report.
    """
    seen: set[str] = set()
    normalized_rows: list[dict[str, Any]] = []
    for index, raw_row in enumerate(rows):
        normalized = _validate_inventory_row(
            raw_row,
            index=index,
            split_ids=split_ids,
            feature_dimension=feature_dimension,
            action_dimension=action_dimension,
        )
        sample_id = normalized["sample_id"]
        if sample_id in seen:
            raise ComparisonPacketError(f"duplicate sample identity: {sample_id}")
        seen.add(sample_id)
        normalized_rows.append(normalized)

    declared_ids = set().union(*split_ids.values())
    if seen != declared_ids:
        missing = sorted(declared_ids - seen)
        extra = sorted(seen - declared_ids)
        raise ComparisonPacketError(
            f"complete sample_inventory does not match declared sample IDs; "
            f"missing={missing}, extra={extra}"
        )
    return {"status": "complete", "row_count": len(normalized_rows), "rows": normalized_rows}


def _validate_row_inventory(
    inventory: Any,
    *,
    split_ids: dict[str, set[str]],
    feature_dimension: int | None,
    action_dimension: int | None,
) -> dict[str, Any]:
    """Validate optional row-level quality, split, finiteness, and action contracts.

    Returns:
        A row-inventory status report and, for complete inventories, normalized rows.
    """
    if inventory is None:
        return {"status": "unavailable", "row_count": None}
    if not isinstance(inventory, dict):
        raise ComparisonPacketError("sample_inventory must be a mapping")
    status = _require_string(inventory, "status", context="sample_inventory")
    if status not in _ALLOWED_INVENTORY_STATES:
        raise ComparisonPacketError(
            f"sample_inventory.status must be one of {sorted(_ALLOWED_INVENTORY_STATES)}"
        )
    rows = inventory.get("rows")
    if status == "complete":
        if not isinstance(rows, list) or not rows:
            raise ComparisonPacketError("complete sample_inventory requires non-empty rows")
    elif rows not in (None, []):
        raise ComparisonPacketError("non-complete sample_inventory must not carry row data")
    if status != "complete":
        return {"status": status, "row_count": None}

    return _validate_complete_inventory(
        rows,
        split_ids=split_ids,
        feature_dimension=feature_dimension,
        action_dimension=action_dimension,
    )


def _validate_artifact_declaration(payload: dict[str, Any]) -> dict[str, str]:
    """Validate the dataset-level artifact declaration.

    Returns:
        Normalized URI, SHA-256 digest, and availability values.
    """
    artifact = payload.get("artifact")
    if not isinstance(artifact, dict):
        raise ComparisonPacketError("artifact must be a mapping")
    artifact_uri = _require_string(artifact, "uri", context="artifact")
    artifact_sha = _require_sha(artifact.get("sha256"), field="artifact.sha256")
    availability = _require_string(artifact, "availability", context="artifact")
    if availability not in {"public", "private_required", "unavailable"}:
        raise ComparisonPacketError("artifact.availability has an unsupported value")
    return {"uri": artifact_uri, "sha256": artifact_sha, "availability": availability}


def _validate_split_ids(
    split: str,
    split_payload: Any,
    *,
    previously_declared: set[str],
) -> set[str]:
    """Validate one split's declared sample identities.

    Returns:
        The unique sample IDs declared for ``split``.
    """
    if not isinstance(split_payload, dict):
        raise ComparisonPacketError(f"splits.{split} must be a mapping")
    raw_ids = split_payload.get("sample_ids")
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ComparisonPacketError(f"splits.{split}.sample_ids must be non-empty")
    ids = {_require_id(sample_id, field=f"splits.{split}.sample_ids") for sample_id in raw_ids}
    if len(ids) != len(raw_ids):
        raise ComparisonPacketError(f"splits.{split}.sample_ids contains duplicates")
    overlap = sorted(previously_declared & ids)
    if overlap:
        if split == "evaluation":
            raise ComparisonPacketError(
                f"holdout contamination: evaluation sample IDs overlap: {overlap}"
            )
        raise ComparisonPacketError(f"split sample IDs overlap: {overlap}")
    return ids


def _validate_shard_location(
    *,
    split: str,
    shard_index: int,
    shard: dict[str, Any],
    root: Path,
    artifact_root: Path | None,
    require_artifacts: bool,
    shard_sha: str,
    size_value: int | None,
) -> dict[str, Any] | None:
    """Validate and optionally verify one shard location.

    Returns:
        Public-safe verification evidence, or ``None`` when URI-only checks are deferred.
    """
    field = f"splits.{split}.shards[{shard_index}]"
    has_path = isinstance(shard.get("path"), str) and bool(shard["path"].strip())
    has_uri = isinstance(shard.get("uri"), str) and bool(shard["uri"].strip())
    if has_path == has_uri:
        raise ComparisonPacketError(f"{field} needs exactly one of path or uri")
    if has_path:
        relative = _safe_relative_path(shard["path"], field=f"{field}.path")
        base = (artifact_root or root).resolve()
        candidate = base / relative
        try:
            candidate.resolve().relative_to(base)
        except ValueError as exc:
            raise ComparisonPacketError(f"{field}.path escapes its root") from exc
        if not require_artifacts:
            return {"status": "not_checked", "path": relative.as_posix()}
        return _check_file(
            candidate,
            expected_sha=shard_sha,
            expected_size=size_value,
            field=f"{split} shard {relative.as_posix()}",
        )

    if not require_artifacts:
        return None
    uri = str(shard["uri"])
    if uri.startswith(_PRIVATE_URI_PREFIX) and artifact_root is not None:
        candidate = _resolve_private_uri(uri, artifact_root, field=f"{field}.uri")
        return _check_file(
            candidate,
            expected_sha=shard_sha,
            expected_size=size_value,
            field=f"{split} private shard",
        )
    raise ComparisonArtifactBlockedError(
        f"{split} shard is unavailable without an authorized artifact root"
    )


def _validate_shard(
    *,
    split: str,
    shard_index: int,
    shard: Any,
    root: Path,
    artifact_root: Path | None,
    require_artifacts: bool,
) -> tuple[set[str], dict[str, Any] | None]:
    """Validate one shard declaration and its sample identities.

    Returns:
        The shard's sample ID set and optional artifact evidence.
    """
    field = f"splits.{split}.shards[{shard_index}]"
    if not isinstance(shard, dict):
        raise ComparisonPacketError(f"{field} must be a mapping")
    shard_sha = _require_sha(shard.get("sha256"), field=f"{field}.sha256")
    shard_sample_ids = shard.get("sample_ids")
    if not isinstance(shard_sample_ids, list) or not shard_sample_ids:
        raise ComparisonPacketError(f"{field}.sample_ids must be non-empty")
    normalized_ids = {
        _require_id(sample_id, field=f"{field}.sample_ids") for sample_id in shard_sample_ids
    }
    if len(normalized_ids) != len(shard_sample_ids):
        raise ComparisonPacketError(f"{field}.sample_ids contains duplicates")
    size_value = shard.get("size_bytes")
    if size_value is not None and (
        isinstance(size_value, bool) or not isinstance(size_value, int) or size_value < 0
    ):
        raise ComparisonPacketError(f"{field}.size_bytes must be a non-negative integer")
    check = _validate_shard_location(
        split=split,
        shard_index=shard_index,
        shard=shard,
        root=root,
        artifact_root=artifact_root,
        require_artifacts=require_artifacts,
        shard_sha=shard_sha,
        size_value=size_value,
    )
    return normalized_ids, check


def _validate_split_shards(
    split: str,
    split_payload: dict[str, Any],
    *,
    declared_ids: set[str],
    root: Path,
    artifact_root: Path | None,
    require_artifacts: bool,
) -> list[dict[str, Any]]:
    """Validate all shard declarations for one split.

    Returns:
        Public-safe artifact checks for shards in this split.
    """
    shards = split_payload.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ComparisonPacketError(f"splits.{split}.shards must be non-empty")
    shard_ids: set[str] = set()
    artifact_checks: list[dict[str, Any]] = []
    for shard_index, shard in enumerate(shards):
        normalized_ids, check = _validate_shard(
            split=split,
            shard_index=shard_index,
            shard=shard,
            root=root,
            artifact_root=artifact_root,
            require_artifacts=require_artifacts,
        )
        overlap = shard_ids & normalized_ids
        if overlap:
            raise ComparisonPacketError(
                f"duplicate sample identity across {split} shards: {sorted(overlap)}"
            )
        shard_ids.update(normalized_ids)
        if check is not None:
            artifact_checks.append(check)
    if shard_ids != declared_ids:
        raise ComparisonPacketError(
            f"splits.{split}.shards sample IDs do not match split declaration; "
            f"missing={sorted(declared_ids - shard_ids)}, extra={sorted(shard_ids - declared_ids)}"
        )
    return artifact_checks


def _validate_splits(
    payload: dict[str, Any],
    *,
    root: Path,
    artifact_root: Path | None,
    require_artifacts: bool,
) -> tuple[dict[str, set[str]], list[dict[str, Any]]]:
    """Validate split identities, shard coverage, and optional artifact bytes.

    Returns:
        A mapping of split IDs and public-safe artifact checks.
    """
    splits = payload.get("splits")
    if not isinstance(splits, dict) or set(splits) != set(SPLITS):
        raise ComparisonPacketError(f"splits must contain exactly {SPLITS}")
    split_ids: dict[str, set[str]] = {}
    artifact_checks: list[dict[str, Any]] = []
    all_ids: set[str] = set()
    for split in SPLITS:
        ids = _validate_split_ids(split, splits[split], previously_declared=all_ids)
        split_ids[split] = ids
        all_ids.update(ids)
        artifact_checks.extend(
            _validate_split_shards(
                split,
                splits[split],
                declared_ids=ids,
                root=root,
                artifact_root=artifact_root,
                require_artifacts=require_artifacts,
            )
        )
    return split_ids, artifact_checks


def _validate_feature_and_action_contracts(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], int | None, int | None]:
    """Validate feature, action, normalization, and row-quality declarations.

    Returns:
        Feature report, action report, and their optional vector dimensions.
    """
    feature_contract = payload.get("feature_contract")
    action_contract = payload.get("action_contract")
    if not isinstance(feature_contract, dict) or not isinstance(action_contract, dict):
        raise ComparisonPacketError("feature_contract and action_contract must be mappings")
    feature_dimension = _validate_contract_dimensions(feature_contract, field="feature_contract")
    action_dimension = _validate_contract_dimensions(action_contract, field="action_contract")
    feature_dtype = _require_string(feature_contract, "dtype", context="feature_contract")
    action_dtype = _require_string(action_contract, "dtype", context="action_contract")
    normalization = payload.get("normalization")
    if not isinstance(normalization, dict):
        raise ComparisonPacketError("normalization must be a mapping")
    _require_string(normalization, "policy", context="normalization")
    quality = payload.get("quality")
    if not isinstance(quality, dict):
        raise ComparisonPacketError("quality must be a mapping")
    if (
        quality.get("admitted_statuses") != [_VALID_ROW_STATUS]
        or not isinstance(quality.get("excluded_statuses"), list)
        or not quality["excluded_statuses"]
    ):
        raise ComparisonPacketError(
            "quality must admit only ['valid'] and declare excluded statuses"
        )
    return (
        {"dtype": feature_dtype, "dimension": feature_dimension},
        {"dtype": action_dtype, "dimension": action_dimension},
        feature_dimension,
        action_dimension,
    )


def _validate_manifest_digest(payload: dict[str, Any]) -> str:
    """Compute and, when present, verify the consumer manifest digest.

    Returns:
        The computed canonical manifest SHA-256 digest.
    """
    computed = _semantic_digest(payload)
    expected = payload.get("manifest_digest")
    if expected is not None and _require_sha(expected, field="manifest_digest") != computed:
        raise ComparisonPacketError(
            "manifest_digest does not match the canonical consumer manifest content"
        )
    return computed


def validate_dataset_consumer_manifest(
    manifest_path: Path,
    *,
    repo_root: Path | None = None,
    artifact_root: Path | None = None,
    require_artifacts: bool = False,
) -> dict[str, Any]:
    """Validate a versioned dataset-consumer manifest.

    Contract-only validation is safe with a private URI.  ``require_artifacts=True`` turns
    missing local shards and digest drift into hard failures before a loader or trainer can be
    constructed.  The returned report contains only manifest-relative names and does not echo
    private absolute paths.

    Returns:
        A public-safe dataset-consumer validation report.
    """
    root = (repo_root or Path.cwd()).resolve()
    path = _resolve_path(manifest_path, root)
    payload = _load_mapping(path)
    if payload.get("schema_version") != CONSUMER_SCHEMA:
        raise ComparisonPacketError(f"schema_version must be {CONSUMER_SCHEMA!r}")
    dataset_id = _require_id(payload.get("dataset_id"), field="dataset_id")
    dataset_version = _require_id(payload.get("dataset_version"), field="dataset_version")
    dataset_digest = _require_sha(payload.get("dataset_digest"), field="dataset_digest")
    artifact = _validate_artifact_declaration(payload)
    split_ids, artifact_checks = _validate_splits(
        payload,
        root=root,
        artifact_root=artifact_root,
        require_artifacts=require_artifacts,
    )
    feature_report, action_report, feature_dimension, action_dimension = (
        _validate_feature_and_action_contracts(payload)
    )
    inventory_report = _validate_row_inventory(
        payload.get("sample_inventory"),
        split_ids=split_ids,
        feature_dimension=feature_dimension,
        action_dimension=action_dimension,
    )
    computed_manifest_digest = _validate_manifest_digest(payload)
    return {
        "status": "verified" if require_artifacts else "declared",
        "schema_version": CONSUMER_SCHEMA,
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "dataset_digest": dataset_digest,
        "manifest_digest": computed_manifest_digest,
        "artifact": artifact,
        "split_counts": {split: len(split_ids[split]) for split in SPLITS},
        "feature_contract": feature_report,
        "action_contract": action_report,
        "sample_inventory": {
            "status": inventory_report["status"],
            "row_count": inventory_report.get("row_count"),
        },
        "artifact_checks": artifact_checks,
        "loader_gate": "fail_closed",
    }


def _require_existing_ref(config: dict[str, Any], key: str, root: Path) -> str:
    """Validate and return a repository-relative config reference.

    Returns:
        The original reference string from the configuration.
    """
    raw = _require_string(config, key, context="comparison")
    path = _resolve_path(raw, root)
    if not path.is_file():
        raise ComparisonPacketError(f"comparison.{key} does not exist: {raw}")
    return raw


def _validate_arm(
    arm_id: str,
    arm: Any,
    *,
    root: Path,
    comparison_budget: int,
    evaluation_seeds: list[int],
    evaluation_scenario: str,
) -> dict[str, Any]:
    """Validate one comparison arm and return its normalized declaration.

    Returns:
        A normalized arm declaration with its config references.
    """
    if not isinstance(arm, dict):
        raise ComparisonPacketError(f"arms.{arm_id} must be a mapping")
    algorithm = _require_string(arm, "algorithm", context=f"arms.{arm_id}").lower()
    initialization = _require_string(arm, "initialization", context=f"arms.{arm_id}").lower()
    if "dagger" in algorithm or "dagger" in initialization:
        raise ComparisonPacketError("DAgger is not admitted in the first comparison packet")
    seeds = _integer_list(arm.get("seeds"), field=f"arms.{arm_id}.seeds")
    env_steps = arm.get("environment_steps")
    if isinstance(env_steps, bool) or not isinstance(env_steps, int) or env_steps <= 0:
        raise ComparisonPacketError(f"arms.{arm_id}.environment_steps must be positive")
    if env_steps != comparison_budget:
        raise ComparisonPacketError(
            f"arms.{arm_id}.environment_steps must equal budget.environment_steps"
        )
    arm_eval_seeds = _integer_list(
        arm.get("evaluation_seeds"), field=f"arms.{arm_id}.evaluation_seeds"
    )
    if arm_eval_seeds != evaluation_seeds:
        raise ComparisonPacketError(
            f"arms.{arm_id}.evaluation_seeds must match the shared held-out evaluation seeds"
        )
    arm_eval_scenario = _require_existing_ref(arm, "evaluation_scenario_config", root)
    if arm_eval_scenario != evaluation_scenario:
        raise ComparisonPacketError(
            f"arms.{arm_id}.evaluation_scenario_config must match the shared evaluation surface"
        )

    config_keys = ["config"]
    if arm_id == "bc_warm_start":
        config_keys.extend(["bc_config", "ppo_config"])
    refs = {key: _require_existing_ref(arm, key, root) for key in config_keys}
    return {
        "arm_id": arm_id,
        "algorithm": algorithm,
        "initialization": initialization,
        "seeds": seeds,
        "environment_steps": env_steps,
        "evaluation_seeds": arm_eval_seeds,
        "evaluation_scenario_config": arm_eval_scenario,
        "configs": refs,
    }


def _validate_comparison_sources(
    config: dict[str, Any],
    *,
    root: Path,
    artifact_root: Path | None,
    require_artifacts: bool,
) -> dict[str, Any]:
    """Validate the canonical launch, trace, split, and consumer sources.

    Returns:
        Source references and their normalized validation reports.
    """
    dataset_manifest_ref = _require_existing_ref(config, "dataset_consumer_manifest", root)
    split_contract = _require_existing_ref(config, "split_contract", root)
    launch_packet_ref = _require_existing_ref(config, "dataset_launch_packet", root)
    trace_registry_ref = _require_existing_ref(config, "trace_uri_registry", root)
    try:
        launch_report = validate_launch_packet(
            _resolve_path(launch_packet_ref, root),
            repo_root=root,
            require_training_ready=False,
        )
    except LaunchPacketError as exc:
        raise ComparisonPacketError(f"dataset launch packet failed validation: {exc}") from exc
    try:
        trace_report = validate_trace_uri_registry(
            _resolve_path(trace_registry_ref, root),
            repo_root=root,
            require_training_ready=False,
        )
    except OracleTraceUriRegistryError as exc:
        raise ComparisonPacketError(f"trace URI registry failed validation: {exc}") from exc
    consumer_report = validate_dataset_consumer_manifest(
        _resolve_path(dataset_manifest_ref, root),
        repo_root=root,
        artifact_root=artifact_root,
        require_artifacts=require_artifacts,
    )
    return {
        "dataset_launch_packet": launch_packet_ref,
        "trace_uri_registry": trace_registry_ref,
        "dataset_consumer_manifest": dataset_manifest_ref,
        "split_contract": split_contract,
        "launch_report": launch_report,
        "trace_report": trace_report,
        "consumer_report": consumer_report,
    }


def _validate_comparison_evaluation(
    config: dict[str, Any],
    *,
    root: Path,
) -> tuple[dict[str, Any], int, list[int], str, dict[str, Any]]:
    """Validate the shared budget and held-out evaluation declaration.

    Returns:
        The budget mapping, environment-step budget, evaluation seeds, scenario reference, and
        normalized evaluation mapping.
    """
    budget = config.get("budget")
    if not isinstance(budget, dict):
        raise ComparisonPacketError("budget must be a mapping")
    comparison_budget = budget.get("environment_steps")
    if (
        isinstance(comparison_budget, bool)
        or not isinstance(comparison_budget, int)
        or comparison_budget <= 0
    ):
        raise ComparisonPacketError("budget.environment_steps must be a positive integer")
    evaluation = config.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ComparisonPacketError("evaluation must be a mapping")
    evaluation_seeds = _integer_list(evaluation.get("seeds"), field="evaluation.seeds")
    evaluation_scenario = _require_existing_ref(evaluation, "scenario_config", root)
    cadence = _integer_list(evaluation.get("cadence_steps"), field="evaluation.cadence_steps")
    if cadence[0] != 0 or cadence[-1] > comparison_budget or cadence != sorted(set(cadence)):
        raise ComparisonPacketError(
            "evaluation.cadence_steps must be sorted, unique, start at 0, and fit the budget"
        )
    checkpoint_selection = evaluation.get("checkpoint_selection")
    if not isinstance(checkpoint_selection, dict):
        raise ComparisonPacketError("evaluation.checkpoint_selection must be a mapping")
    if checkpoint_selection.get("validation_split") != "validation":
        raise ComparisonPacketError(
            "checkpoint_selection.validation_split must be the validation split"
        )
    _require_string(checkpoint_selection, "metric", context="checkpoint_selection")
    _require_string(checkpoint_selection, "direction", context="checkpoint_selection")
    if checkpoint_selection.get("retained") != ["best_validation", "final"]:
        raise ComparisonPacketError(
            "checkpoint_selection.retained must be ['best_validation', 'final']"
        )
    evaluation["seeds"] = evaluation_seeds
    evaluation["scenario_config"] = evaluation_scenario
    evaluation["cadence_steps"] = cadence
    evaluation["checkpoint_selection"] = checkpoint_selection
    return budget, comparison_budget, evaluation_seeds, evaluation_scenario, evaluation


def _validate_comparison_arms(
    config: dict[str, Any],
    *,
    root: Path,
    comparison_budget: int,
    evaluation_seeds: list[int],
    evaluation_scenario: str,
) -> dict[str, dict[str, Any]]:
    """Validate both comparison arms and their matched randomization.

    Returns:
        Normalized RL-only and BC-warm-start arm declarations.
    """
    arms_payload = config.get("arms")
    if not isinstance(arms_payload, dict) or set(arms_payload) != {
        "rl_only",
        "bc_warm_start",
    }:
        raise ComparisonPacketError("arms must contain exactly rl_only and bc_warm_start")
    arms = {
        arm_id: _validate_arm(
            arm_id,
            arms_payload[arm_id],
            root=root,
            comparison_budget=comparison_budget,
            evaluation_seeds=evaluation_seeds,
            evaluation_scenario=evaluation_scenario,
        )
        for arm_id in ("rl_only", "bc_warm_start")
    }
    if arms["rl_only"]["seeds"] != arms["bc_warm_start"]["seeds"]:
        raise ComparisonPacketError("RL-only and BC-warm-start arms must use matched seeds")
    return arms


def _validate_estimands_and_scope(config: dict[str, Any]) -> None:
    """Validate pending estimands and ensure this packet cannot execute research."""
    estimands = config.get("estimands")
    if not isinstance(estimands, dict):
        raise ComparisonPacketError("estimands must be a mapping")
    for name in ("sample_efficiency", "final_performance"):
        section = estimands.get(name)
        if not isinstance(section, dict):
            raise ComparisonPacketError(f"estimands.{name} must be a mapping")
        _require_string(section, "definition", context=f"estimands.{name}")
        if section.get("result_status") != "pending":
            raise ComparisonPacketError(f"estimands.{name}.result_status must be pending")
    out_of_scope = config.get("out_of_scope_actions")
    if not isinstance(out_of_scope, dict):
        raise ComparisonPacketError("out_of_scope_actions must be a mapping")
    required_false = {
        "data_collection",
        "dagger_refinement",
        "training_execution",
        "benchmark_execution",
        "publication_claim",
    }
    if any(out_of_scope.get(key) is not False for key in required_false):
        raise ComparisonPacketError("comparison packet out_of_scope_actions must remain false")


def build_comparison_packet(
    config_path: Path,
    *,
    repo_root: Path | None = None,
    artifact_root: Path | None = None,
    require_artifacts: bool = False,
) -> dict[str, Any]:
    """Build the deterministic comparison packet without executing training.

    Returns:
        A ready packet containing source provenance, matched arms, and claim boundaries.
    """
    root = (repo_root or Path.cwd()).resolve()
    path = _resolve_path(config_path, root)
    config = _load_mapping(path)
    if config.get("schema_version") != COMPARISON_SCHEMA:
        raise ComparisonPacketError(f"schema_version must be {COMPARISON_SCHEMA!r}")
    experiment_id = _require_id(config.get("experiment_id"), field="experiment_id")
    sources = _validate_comparison_sources(
        config,
        root=root,
        artifact_root=artifact_root,
        require_artifacts=require_artifacts,
    )
    budget, comparison_budget, evaluation_seeds, evaluation_scenario, evaluation = (
        _validate_comparison_evaluation(config, root=root)
    )
    arms = _validate_comparison_arms(
        config,
        root=root,
        comparison_budget=comparison_budget,
        evaluation_seeds=evaluation_seeds,
        evaluation_scenario=evaluation_scenario,
    )
    _validate_estimands_and_scope(config)
    estimands = config["estimands"]
    cadence = evaluation["cadence_steps"]
    checkpoint_selection = evaluation["checkpoint_selection"]

    identities: list[dict[str, str | int]] = []
    for arm_id in ("rl_only", "bc_warm_start"):
        for seed in arms[arm_id]["seeds"]:
            run_id = f"{experiment_id}__{arm_id}__seed{seed}"
            identities.append(
                {
                    "arm_id": arm_id,
                    "seed": seed,
                    "run_id": run_id,
                    "checkpoint_id": f"{run_id}__checkpoint__best_validation",
                    "evaluation_id": f"{run_id}__evaluation__holdout",
                }
            )

    return {
        "schema_version": COMPARISON_SCHEMA,
        "status": "ready",
        "experiment_id": experiment_id,
        "source": {
            "dataset_launch_packet": sources["dataset_launch_packet"],
            "trace_uri_registry": sources["trace_uri_registry"],
            "dataset_consumer_manifest": sources["dataset_consumer_manifest"],
            "split_contract": sources["split_contract"],
            "dataset_id": sources["launch_report"]["dataset_id"],
            "trace_dataset_id": sources["trace_report"]["dataset_id"],
        },
        "dataset": sources["consumer_report"],
        "budget": {
            "environment_steps": comparison_budget,
            "comparison_scope": _require_string(budget, "comparison_scope", context="budget"),
            "initialization_cost_separate": budget.get("initialization_cost_separate") is True,
        },
        "arms": arms,
        "evaluation": {
            "seeds": evaluation_seeds,
            "scenario_config": evaluation_scenario,
            "cadence_steps": cadence,
            "checkpoint_selection": checkpoint_selection,
        },
        "estimands": estimands,
        "identities": identities,
        "loader_gate": {
            "status": sources["consumer_report"]["status"],
            "requires_artifacts": require_artifacts,
            "failure_policy": "fail_closed_before_model_construction",
            "checks": [
                "artifact_digest_and_size",
                "schema_and_split_identity",
                "holdout_contamination",
                "row_quality_and_finiteness",
                "observation_action_contract",
                "duplicate_sample_identity",
            ],
        },
        "startup_canary": {
            "status": "available_with_synthetic_or_disjoint_inventory",
            "training_budget": "one_bounded_optimization_update",
            "nominal_dataset_access": False,
            "benchmark_evidence": False,
        },
        "out_of_scope_actions": dict.fromkeys(
            sorted(
                {
                    "data_collection",
                    "dagger_refinement",
                    "training_execution",
                    "benchmark_execution",
                    "publication_claim",
                }
            ),
            False,
        ),
        "claim_boundary": (
            "Preparation and startup-canary wiring only; no nominal training, benchmark outcome, "
            "sample-efficiency result, final-performance result, or publication claim."
        ),
    }


def run_startup_canary(
    manifest_path: Path,
    output_dir: Path,
    *,
    repo_root: Path | None = None,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    """Run one tiny synthetic/disjoint loader-model-checkpoint-evaluation canary.

    Returns:
        A diagnostic canary report that is explicitly not benchmark evidence.
    """
    report = validate_dataset_consumer_manifest(
        manifest_path,
        repo_root=repo_root,
        artifact_root=artifact_root,
        require_artifacts=True,
    )
    payload = _load_mapping(_resolve_path(manifest_path, (repo_root or Path.cwd()).resolve()))
    inventory = payload.get("sample_inventory")
    if not isinstance(inventory, dict) or inventory.get("status") != "complete":
        raise ComparisonPacketError(
            "startup canary requires a complete synthetic or disjoint sample inventory"
        )
    rows = inventory.get("rows")
    if not isinstance(rows, list):
        raise ComparisonPacketError("startup canary sample inventory rows are unavailable")
    train_rows = [row for row in rows if row.get("split") == "train"]
    eval_rows = [row for row in rows if row.get("split") == "evaluation"]
    if not train_rows or not eval_rows:
        raise ComparisonPacketError("startup canary requires train and evaluation rows")
    features = np.asarray([row["features"] for row in train_rows], dtype=np.float32)
    actions = np.asarray([row["action"] for row in train_rows], dtype=np.float32)
    eval_features = np.asarray([row["features"] for row in eval_rows], dtype=np.float32)
    eval_actions = np.asarray([row["action"] for row in eval_rows], dtype=np.float32)
    weights = np.zeros((features.shape[1], actions.shape[1]), dtype=np.float32)
    initial_loss = float(np.mean((features @ weights - actions) ** 2))
    gradient = (2.0 / len(features)) * features.T @ (features @ weights - actions)
    weights -= np.float32(0.1) * gradient.astype(np.float32)
    post_update_loss = float(np.mean((features @ weights - actions) ** 2))
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "startup_canary_checkpoint.npz"
    np.savez(checkpoint_path, weights=weights)
    with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
        restored = np.asarray(checkpoint["weights"], dtype=np.float32)
    if not np.array_equal(restored, weights):
        raise ComparisonPacketError("startup canary checkpoint round-trip changed model weights")
    evaluation_loss = float(np.mean((eval_features @ restored - eval_actions) ** 2))
    result = {
        "schema_version": CANARY_SCHEMA,
        "status": "passed",
        "evidence_tier": "startup_canary",
        "training_executed": True,
        "benchmark_evidence": False,
        "num_updates": 1,
        "train_rows": len(train_rows),
        "evaluation_rows": len(eval_rows),
        "initial_loss": initial_loss,
        "post_update_loss": post_update_loss,
        "evaluation_startup_loss": evaluation_loss,
        "checkpoint_round_trip": "passed",
        "checkpoint_file": checkpoint_path.name,
        "dataset_manifest_digest": report["manifest_digest"],
        "claim_boundary": (
            "Synthetic/disjoint wiring only; loss values are canary diagnostics and are not "
            "BC quality, benchmark, or publication evidence."
        ),
    }
    (output_dir / "startup_canary_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


__all__ = [
    "CANARY_SCHEMA",
    "COMPARISON_SCHEMA",
    "CONSUMER_SCHEMA",
    "ComparisonArtifactBlockedError",
    "ComparisonPacketError",
    "build_comparison_packet",
    "run_startup_canary",
    "validate_dataset_consumer_manifest",
]
