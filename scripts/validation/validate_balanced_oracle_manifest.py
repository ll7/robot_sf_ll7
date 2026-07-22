#!/usr/bin/env python3
"""Validation CLI for balanced oracle dataset manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
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

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to parse manifest JSON: {exc}") from exc

    if not isinstance(manifest, dict):
        raise ValueError("Manifest payload must be a JSON object")

    errors: list[str] = []

    schema_version = manifest.get("schema_version")
    if schema_version != EXPECTED_SCHEMA:
        errors.append(f"schema_version must be {EXPECTED_SCHEMA!r}, got {schema_version!r}")

    dataset_id = manifest.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        errors.append("dataset_id must be a non-empty string")

    exact_public_sha = manifest.get("exact_public_sha")
    if not isinstance(exact_public_sha, str) or not exact_public_sha.strip():
        errors.append("exact_public_sha must be a non-empty string")

    sha256_inv = manifest.get("sha256_inventory")
    if not isinstance(sha256_inv, dict) or not sha256_inv:
        errors.append("sha256_inventory must be a non-empty mapping")

    exclusions = manifest.get("exclusions")
    if not isinstance(exclusions, list):
        errors.append("exclusions must be a list")

    balance_summary = manifest.get("balance_summary")
    if not isinstance(balance_summary, dict):
        errors.append("balance_summary must be a mapping")
    else:
        if "action_bin_accounting" not in balance_summary:
            errors.append("balance_summary is missing action_bin_accounting")
        if "stratum_counts" not in balance_summary:
            errors.append("balance_summary is missing stratum_counts")

    registry_candidate = manifest.get("private_artifact_registry_candidate")
    if not isinstance(registry_candidate, dict):
        errors.append("private_artifact_registry_candidate must be a mapping")

    bc_smoke_cmd = manifest.get("bc_loader_smoke_command")
    if not isinstance(bc_smoke_cmd, str) or not bc_smoke_cmd.strip():
        errors.append("bc_loader_smoke_command must be a non-empty string")

    # Check NPZ file presence and SHA-256 match
    npz_path_raw = manifest.get("npz_path")
    npz_path = Path(npz_path_raw) if npz_path_raw else manifest_path.parent / "expert_traj_v1.npz"
    if not npz_path.is_absolute():
        npz_path = (manifest_path.parent / npz_path).resolve()

    if not npz_path.is_file():
        errors.append(f"NPZ artifact is missing: {npz_path}")
    else:
        actual_sha = _file_sha256(npz_path)
        if exact_public_sha and actual_sha != exact_public_sha:
            errors.append(f"NPZ SHA-256 mismatch: expected {exact_public_sha}, got {actual_sha}")

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
    except Exception as exc:
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
