"""Manifest contracts for standalone offline SAC pretraining."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from gymnasium import spaces as gym_spaces

from robot_sf.benchmark.rl_trajectory_dataset import (
    RL_TRAJECTORY_DATASET_SCHEMA_VERSION,
    sha256_file,
)

if TYPE_CHECKING:
    from robot_sf.training.offline_online_rl import OfflineTransitionBatch

OFFLINE_POLICY_CHECKPOINT_MANIFEST_SCHEMA_VERSION = "offline-policy-checkpoint-manifest.v1"
OFFLINE_TO_ONLINE_FINETUNE_MANIFEST_SCHEMA_VERSION = "offline-to-online-finetune-manifest.v1"
NORMALIZER_STATE_SCHEMA_VERSION = "offline-pretraining-normalizer-state.v1"
CLAIM_BOUNDARY = "standalone offline pretraining provenance only; not benchmark evidence"
FINETUNE_CLAIM_BOUNDARY = (
    "diagnostic offline-to-online fine-tune provenance only; not benchmark evidence"
)


def utc_now_iso() -> str:
    """Return a stable UTC timestamp string for manifests."""

    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def current_git_commit() -> str:
    """Return current git commit or ``unknown`` outside a git checkout."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write deterministic JSON and create parent directories."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_normalizer_state(path: Path, *, present: bool, reason: str) -> dict[str, Any]:
    """Write a hashed normalizer sidecar for VecNormalize or explicit absence.

    Returns:
        JSON payload written to disk.
    """

    payload = {
        "schema_version": NORMALIZER_STATE_SCHEMA_VERSION,
        "present": bool(present),
        "reason": reason,
        "created_at_utc": utc_now_iso(),
    }
    write_json(path, payload)
    return payload


def space_fingerprint(space: gym_spaces.Space[Any]) -> str:
    """Return a SHA-256 fingerprint for a Gymnasium space contract."""

    payload = _space_payload(space)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def offline_dataset_manifest_summary(
    *,
    dataset_manifest_path: Path,
    dataset_path: Path,
    batch: OfflineTransitionBatch,
) -> dict[str, Any]:
    """Build the dataset section used by checkpoint manifests.

    Returns:
        Manifest-ready dataset provenance block.
    """

    if not dataset_manifest_path.is_file():
        raise ValueError(f"offline dataset manifest not found: {dataset_manifest_path}")
    manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    dataset_id = str(manifest.get("dataset_id") or "")
    if not dataset_id:
        raise ValueError("offline dataset manifest missing dataset_id")
    return {
        "dataset_path": str(dataset_path),
        "dataset_manifest_path": str(dataset_manifest_path),
        "dataset_sha256": sha256_file(dataset_path),
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "dataset_schema_version": RL_TRAJECTORY_DATASET_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "split": batch.preflight.split,
        "episode_count": batch.preflight.episode_count,
        "accepted_transitions": batch.preflight.accepted_transitions,
        "dropped_terminal_transitions": batch.preflight.dropped_terminal_transitions,
    }


def build_offline_checkpoint_manifest(
    *,
    checkpoint_path: Path,
    normalizer_path: Path,
    training_config_path: Path,
    dataset: Mapping[str, Any],
    offline_training: Mapping[str, Any],
    environment_contract: Mapping[str, Any],
    policy_type: str,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build and validate an offline checkpoint manifest.

    Returns:
        Complete manifest payload.
    """

    manifest = {
        "schema_version": OFFLINE_POLICY_CHECKPOINT_MANIFEST_SCHEMA_VERSION,
        "issue": 4245,
        "parent_issue": 4012,
        "claim_boundary": CLAIM_BOUNDARY,
        "algorithm": "sac",
        "policy_type": policy_type,
        "checkpoint_path": str(checkpoint_path),
        "normalizer_path": str(normalizer_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "normalizer_sha256": sha256_file(normalizer_path),
        "training_config_path": str(training_config_path),
        "training_config_sha256": sha256_file(training_config_path),
        "dataset": dict(dataset),
        "offline_training": dict(offline_training),
        "environment_contract": dict(environment_contract),
        "created_at_utc": created_at_utc or utc_now_iso(),
        "git_commit": current_git_commit(),
        "eligible_for_claim": False,
    }
    validate_offline_checkpoint_manifest(manifest)
    return manifest


def build_finetune_manifest(  # noqa: PLR0913
    *,
    parent_manifest_path: Path,
    parent_manifest: Mapping[str, Any],
    checkpoint_path: Path,
    normalizer_path: Path,
    training_config_path: Path,
    online_timesteps: int,
    seed: int | None,
    environment_contract: Mapping[str, Any],
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build and validate an offline-to-online fine-tune manifest.

    Returns:
        Complete fine-tune manifest payload.
    """

    manifest = {
        "schema_version": OFFLINE_TO_ONLINE_FINETUNE_MANIFEST_SCHEMA_VERSION,
        "issue": 4245,
        "parent_issue": 4012,
        "claim_boundary": FINETUNE_CLAIM_BOUNDARY,
        "algorithm": "sac",
        "parent_offline_checkpoint_manifest_path": str(parent_manifest_path),
        "parent_offline_checkpoint_manifest_sha256": sha256_file(parent_manifest_path),
        "parent_checkpoint_sha256": str(parent_manifest["checkpoint_sha256"]),
        "parent_normalizer_sha256": str(parent_manifest["normalizer_sha256"]),
        "inherited_dataset": dict(parent_manifest["dataset"]),
        "checkpoint_path": str(checkpoint_path),
        "normalizer_path": str(normalizer_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "normalizer_sha256": sha256_file(normalizer_path),
        "online_finetune_config_path": str(training_config_path),
        "online_finetune_config_sha256": sha256_file(training_config_path),
        "online_timesteps": int(online_timesteps),
        "seed": seed,
        "environment_contract": dict(environment_contract),
        "created_at_utc": created_at_utc or utc_now_iso(),
        "git_commit": current_git_commit(),
        "eligible_for_claim": False,
    }
    validate_finetune_manifest(manifest)
    return manifest


def load_offline_checkpoint_manifest(path: Path | str) -> dict[str, Any]:
    """Load and validate an offline checkpoint manifest from disk.

    Returns:
        Parsed manifest payload.
    """

    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("offline checkpoint manifest must be a JSON object")
    validate_offline_checkpoint_manifest(manifest, base_dir=manifest_path.parent)
    return manifest


def validate_offline_checkpoint_manifest(
    manifest: Mapping[str, Any],
    *,
    base_dir: Path | None = None,
) -> None:
    """Fail closed when required offline checkpoint provenance is incomplete."""

    _require_schema(manifest, OFFLINE_POLICY_CHECKPOINT_MANIFEST_SCHEMA_VERSION)
    _require_fields(
        manifest,
        (
            "checkpoint_path",
            "normalizer_path",
            "checkpoint_sha256",
            "normalizer_sha256",
            "training_config_path",
            "training_config_sha256",
            "dataset",
            "offline_training",
            "environment_contract",
        ),
    )
    dataset = _mapping_field(manifest, "dataset")
    _require_fields(
        dataset,
        (
            "dataset_path",
            "dataset_manifest_path",
            "dataset_sha256",
            "dataset_manifest_sha256",
            "dataset_schema_version",
            "dataset_id",
            "split",
            "episode_count",
            "accepted_transitions",
        ),
    )
    if dataset["dataset_schema_version"] != RL_TRAJECTORY_DATASET_SCHEMA_VERSION:
        raise ValueError("offline checkpoint manifest dataset schema mismatch")
    _validate_environment_contract(_mapping_field(manifest, "environment_contract"))
    _verify_file_sha(manifest, "checkpoint_path", "checkpoint_sha256", base_dir=base_dir)
    _verify_file_sha(manifest, "normalizer_path", "normalizer_sha256", base_dir=base_dir)
    _verify_file_sha(manifest, "training_config_path", "training_config_sha256", base_dir=base_dir)
    _verify_file_sha(dataset, "dataset_path", "dataset_sha256", base_dir=base_dir)
    _verify_file_sha(dataset, "dataset_manifest_path", "dataset_manifest_sha256", base_dir=base_dir)


def validate_finetune_manifest(
    manifest: Mapping[str, Any],
    *,
    base_dir: Path | None = None,
) -> None:
    """Fail closed when fine-tune provenance chain is incomplete."""

    _require_schema(manifest, OFFLINE_TO_ONLINE_FINETUNE_MANIFEST_SCHEMA_VERSION)
    _require_fields(
        manifest,
        (
            "parent_offline_checkpoint_manifest_path",
            "parent_offline_checkpoint_manifest_sha256",
            "parent_checkpoint_sha256",
            "parent_normalizer_sha256",
            "inherited_dataset",
            "checkpoint_path",
            "normalizer_path",
            "checkpoint_sha256",
            "normalizer_sha256",
            "online_finetune_config_path",
            "online_finetune_config_sha256",
            "online_timesteps",
            "environment_contract",
        ),
    )
    _validate_environment_contract(_mapping_field(manifest, "environment_contract"))
    _require_fields(_mapping_field(manifest, "inherited_dataset"), ("dataset_id", "dataset_sha256"))
    _verify_file_sha(
        manifest,
        "parent_offline_checkpoint_manifest_path",
        "parent_offline_checkpoint_manifest_sha256",
        base_dir=base_dir,
    )
    _verify_file_sha(manifest, "checkpoint_path", "checkpoint_sha256", base_dir=base_dir)
    _verify_file_sha(manifest, "normalizer_path", "normalizer_sha256", base_dir=base_dir)
    _verify_file_sha(
        manifest,
        "online_finetune_config_path",
        "online_finetune_config_sha256",
        base_dir=base_dir,
    )
    _validate_finetune_normalizer_compatibility(manifest, base_dir=base_dir)


def _validate_finetune_normalizer_compatibility(
    manifest: Mapping[str, Any], *, base_dir: Path | None
) -> None:
    parent_state = _load_normalizer_state(
        manifest["parent_offline_checkpoint_manifest_path"],
        normalizer_path_key="normalizer_path",
        base_dir=base_dir,
    )
    parent_present = bool(parent_state.get("present"))
    if not parent_present:
        return

    current_state = _load_normalizer_state(
        manifest,
        normalizer_path_key="normalizer_path",
        base_dir=base_dir,
    )
    current_parent_sha = str(manifest["parent_normalizer_sha256"])
    if (
        not bool(current_state.get("present"))
        or str(manifest["normalizer_sha256"]) != current_parent_sha
    ):
        raise ValueError(
            "fine-tune manifest must apply matching parent normalizer state "
            "when the parent offline checkpoint requires one"
        )


def _load_normalizer_state(
    manifest_or_path: Mapping[str, Any] | Path | str,
    *,
    normalizer_path_key: str,
    base_dir: Path | None,
) -> dict[str, Any]:
    if isinstance(manifest_or_path, Mapping):
        manifest = manifest_or_path
        manifest_dir = base_dir
    else:
        manifest_path = _resolve_manifest_path(Path(manifest_or_path), base_dir=base_dir)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("parent offline checkpoint manifest must JSON object")
        manifest_dir = manifest_path.parent

    normalizer_path = Path(str(manifest[normalizer_path_key]))
    if not normalizer_path.is_absolute() and manifest_dir is not None:
        normalizer_path = manifest_dir / normalizer_path
    try:
        payload = json.loads(normalizer_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"normalizer state must be readable JSON: {normalizer_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"normalizer state must be JSON object: {normalizer_path}")
    if payload.get("schema_version") != NORMALIZER_STATE_SCHEMA_VERSION:
        raise ValueError(f"normalizer state schema mismatch: {normalizer_path}")
    return payload


def _resolve_manifest_path(path: Path, *, base_dir: Path | None) -> Path:
    if not path.is_absolute() and base_dir is not None:
        return base_dir / path
    return path


def assert_environment_compatible(
    *,
    parent_contract: Mapping[str, Any],
    current_contract: Mapping[str, Any],
) -> None:
    """Require exact observation/action fingerprint compatibility."""

    for key in ("observation_space_fingerprint", "action_space_fingerprint"):
        if parent_contract.get(key) != current_contract.get(key):
            raise ValueError(
                f"offline checkpoint environment {key} mismatch: "
                f"{parent_contract.get(key)!r} != {current_contract.get(key)!r}"
            )


def _space_payload(space: gym_spaces.Space[Any]) -> dict[str, Any]:
    if isinstance(space, gym_spaces.Box):
        return {
            "type": "Box",
            "shape": tuple(int(value) for value in space.shape),
            "dtype": str(space.dtype),
            "low": _array_summary(space.low),
            "high": _array_summary(space.high),
        }
    if isinstance(space, gym_spaces.Dict):
        return {
            "type": "Dict",
            "spaces": {key: _space_payload(value) for key, value in sorted(space.spaces.items())},
        }
    if isinstance(space, gym_spaces.Discrete):
        return {"type": "Discrete", "n": int(space.n), "start": int(space.start)}
    return {"type": type(space).__name__, "repr": repr(space)}


def _array_summary(value: np.ndarray) -> Any:
    array = np.asarray(value)
    if array.size <= 16:
        return array.tolist()
    return {
        "shape": tuple(int(dim) for dim in array.shape),
        "dtype": str(array.dtype),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _require_schema(manifest: Mapping[str, Any], expected: str) -> None:
    if manifest.get("schema_version") != expected:
        raise ValueError(
            f"expected schema_version {expected!r}, got {manifest.get('schema_version')!r}"
        )


def _require_fields(mapping: Mapping[str, Any], fields: tuple[str, ...]) -> None:
    missing = [field for field in fields if mapping.get(field) in (None, "")]
    if missing:
        raise ValueError(f"manifest missing required fields: {missing}")


def _mapping_field(mapping: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = mapping.get(field)
    if not isinstance(value, Mapping):
        raise ValueError(f"manifest field {field!r} must be an object")
    return value


def _validate_environment_contract(contract: Mapping[str, Any]) -> None:
    _require_fields(
        contract,
        ("scenario_config", "observation_space_fingerprint", "action_space_fingerprint"),
    )


def _verify_file_sha(
    mapping: Mapping[str, Any],
    path_key: str,
    sha_key: str,
    *,
    base_dir: Path | None,
) -> None:
    path = Path(str(mapping[path_key]))
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    if not path.is_file():
        raise ValueError(f"manifest file not found for {path_key}: {path}")
    actual = sha256_file(path)
    expected = str(mapping[sha_key])
    if actual != expected:
        raise ValueError(f"manifest checksum mismatch for {path_key}: {actual} != {expected}")
