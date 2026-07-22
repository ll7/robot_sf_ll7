#!/usr/bin/env python3
"""Validation CLI for balanced oracle dataset manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

EXPECTED_SCHEMA = "balanced-oracle-dataset-manifest.v1"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load a manifest and require a JSON object payload."""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed to parse manifest JSON: {exc}") from exc

    if not isinstance(manifest, dict):
        raise ValueError("Manifest payload must be a JSON object")
    return manifest


def _validate_balance_summary(manifest: dict[str, Any], errors: list[str]) -> None:
    """Append validation errors for the manifest's balance summary."""
    balance_summary = manifest.get("balance_summary")
    if not isinstance(balance_summary, dict):
        errors.append("balance_summary must be a mapping")
        return

    for required_key in ("action_bin_accounting", "stratum_counts"):
        if required_key not in balance_summary:
            errors.append(f"balance_summary is missing {required_key}")
    transitions = balance_summary.get("usable_train_transitions")
    episodes = balance_summary.get("usable_train_episodes")
    if not isinstance(transitions, int) or transitions <= 0:
        errors.append("balance_summary.usable_train_transitions must be positive")
    if not isinstance(episodes, int) or episodes <= 0:
        errors.append("balance_summary.usable_train_episodes must be positive")

    action_accounting = balance_summary.get("action_bin_accounting")
    if not isinstance(action_accounting, dict):
        errors.append("balance_summary.action_bin_accounting must be a mapping")
    elif action_accounting.get("total_transitions") != transitions:
        errors.append("action-bin transition count does not match usable training transitions")
    elif not isinstance(action_accounting.get("weights_sha256"), str):
        errors.append("action-bin accounting must include weights_sha256")


def _validate_manifest_fields(manifest: dict[str, Any]) -> list[str]:  # noqa: C901, PLR0912
    """Return errors for required manifest metadata."""
    errors: list[str] = []

    schema_version = manifest.get("schema_version")
    if schema_version != EXPECTED_SCHEMA:
        errors.append(f"schema_version must be {EXPECTED_SCHEMA!r}, got {schema_version!r}")

    for field_name in (
        "dataset_id",
        "exact_public_sha",
        "dataset_sha256",
        "bc_loader_smoke_command",
    ):
        value = manifest.get(field_name)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{field_name} must be a non-empty string")

    sha256_inventory = manifest.get("sha256_inventory")
    if not isinstance(sha256_inventory, dict) or not sha256_inventory:
        errors.append("sha256_inventory must be a non-empty mapping")

    if not isinstance(manifest.get("exclusions"), list):
        errors.append("exclusions must be a list")

    exact_public_sha = manifest.get("exact_public_sha")
    if (
        isinstance(exact_public_sha, str)
        and re.fullmatch(r"[0-9a-f]{40}", exact_public_sha) is None
    ):
        errors.append("exact_public_sha must be a 40-character Git SHA")
    if exact_public_sha != manifest.get("git_commit"):
        errors.append("exact_public_sha must match git_commit")
    if (
        isinstance(sha256_inventory, dict)
        and "raw_episode_provenance.jsonl" not in sha256_inventory
    ):
        errors.append("sha256_inventory must include raw_episode_provenance.jsonl")

    if manifest.get("eligibility_status") != "training_ready":
        errors.append("eligibility_status must be 'training_ready'")
    yield_gates = manifest.get("yield_gates")
    if not isinstance(yield_gates, dict) or yield_gates.get("status") != "pass":
        errors.append("yield_gates.status must be 'pass'")
    elif isinstance(manifest.get("balance_summary"), dict):
        summary = manifest["balance_summary"]
        min_transitions = yield_gates.get("min_usable_transitions")
        observed_transitions = summary.get("usable_train_transitions")
        if not isinstance(min_transitions, int) or not isinstance(observed_transitions, int):
            errors.append("yield transition gate values must be integers")
        elif observed_transitions < min_transitions:
            errors.append("usable training transitions do not satisfy the declared yield gate")
        min_per_stratum = yield_gates.get("min_episodes_per_stratum")
        stratum_counts = summary.get("stratum_counts", {}).get("train", {})
        if not isinstance(min_per_stratum, int) or not isinstance(stratum_counts, dict):
            errors.append("per-stratum yield gate values are invalid")
        else:
            inadequate = {
                stratum: count
                for stratum, count in stratum_counts.items()
                if not isinstance(count, int) or count < min_per_stratum
            }
            if inadequate:
                errors.append(f"training strata do not satisfy the yield gate: {inadequate}")
    if manifest.get("missing_episode_ids") not in ([], None):
        errors.append("missing_episode_ids must be empty")

    registry_candidate = manifest.get("private_artifact_registry_candidate")
    if not isinstance(registry_candidate, dict):
        errors.append("private_artifact_registry_candidate must be a mapping")
    else:
        if registry_candidate.get("sha256") != manifest.get("dataset_sha256"):
            errors.append("registry candidate SHA-256 must match dataset_sha256")
        splits = registry_candidate.get("splits")
        if not isinstance(splits, dict):
            errors.append("private_artifact_registry_candidate.splits must be a mapping")
        else:
            for split in ("train", "validation", "evaluation"):
                split_payload = splits.get(split)
                episode_ids = (
                    split_payload.get("episode_ids") if isinstance(split_payload, dict) else None
                )
                if not isinstance(episode_ids, list) or not episode_ids:
                    errors.append(f"registry candidate split {split!r} must be non-empty")

    _validate_balance_summary(manifest, errors)
    return errors


def _resolve_npz_path(manifest_path: Path, manifest: dict[str, Any]) -> Path:
    """Resolve the artifact path relative to its manifest when necessary."""
    npz_path_raw = manifest.get("npz_path")
    if npz_path_raw:
        try:
            npz_path = Path(npz_path_raw)
        except TypeError as exc:
            raise ValueError("npz_path must be a string") from exc
    else:
        npz_path = manifest_path.parent / "expert_traj_v1.npz"
    if npz_path_raw and not npz_path.is_absolute():
        npz_path = manifest_path.parent / npz_path
    return npz_path.resolve()


def _validate_npz_artifact(npz_path: Path, expected_sha: Any, errors: list[str]) -> None:
    """Append validation errors for artifact presence and digest integrity."""
    if not npz_path.is_file():
        errors.append(f"NPZ artifact is missing: {npz_path}")
        return

    actual_sha = _file_sha256(npz_path)
    if expected_sha and actual_sha != expected_sha:
        errors.append(f"NPZ SHA-256 mismatch: expected {expected_sha}, got {actual_sha}")


def _validate_inventory(manifest_path: Path, manifest: dict[str, Any], errors: list[str]) -> None:
    inventory = manifest.get("sha256_inventory")
    if not isinstance(inventory, dict):
        return
    for relative_name, expected_sha in inventory.items():
        if not isinstance(relative_name, str) or not isinstance(expected_sha, str):
            errors.append("sha256_inventory entries must map paths to SHA-256 strings")
            continue
        artifact_path = (manifest_path.parent / relative_name).resolve()
        try:
            artifact_path.relative_to(manifest_path.parent.resolve())
        except ValueError:
            errors.append(f"inventory path escapes the manifest directory: {relative_name}")
            continue
        if not artifact_path.is_file():
            errors.append(f"inventory artifact is missing: {artifact_path}")
            continue
        actual_sha = _file_sha256(artifact_path)
        if actual_sha != expected_sha:
            errors.append(
                f"inventory SHA-256 mismatch for {relative_name}: "
                f"expected {expected_sha}, got {actual_sha}"
            )


def validate_balanced_manifest(manifest_path: Path) -> dict[str, Any]:
    """Validate a balanced oracle dataset manifest and associated NPZ artifact.

    Args:
        manifest_path: Path to balanced_oracle_dataset_manifest.json.

    Returns:
        Validation report.

    Raises:
        ValueError: If manifest or artifact fails validation.
    """
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise ValueError(f"Manifest path is not a file: {manifest_path}")

    manifest = _load_manifest(manifest_path)
    errors = _validate_manifest_fields(manifest)
    dataset_id = manifest.get("dataset_id")
    exact_public_sha = manifest.get("exact_public_sha")
    exclusions = manifest.get("exclusions")
    npz_path = _resolve_npz_path(manifest_path, manifest)
    _validate_npz_artifact(npz_path, manifest.get("dataset_sha256"), errors)
    _validate_inventory(manifest_path, manifest, errors)

    if errors:
        raise ValueError("Balanced manifest validation failed:\n- " + "\n- ".join(errors))

    return {
        "status": "valid",
        "manifest_path": str(manifest_path),
        "dataset_id": dataset_id,
        "exact_public_sha": exact_public_sha,
        "npz_path": str(npz_path),
        "exclusions_count": len(exclusions) if isinstance(exclusions, list) else 0,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_path", type=Path, help="Path to manifest JSON file.")
    parser.add_argument("--json", action="store_true", help="Output validation report as JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        report = validate_balanced_manifest(args.manifest_path)
    except (OSError, ValueError) as exc:
        if args.json:
            print(json.dumps({"status": "invalid", "error": str(exc)}, indent=2))
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Manifest is valid: {report['manifest_path']}")
        print(f"Dataset ID: {report['dataset_id']}")
        print(f"SHA-256: {report['exact_public_sha']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
