#!/usr/bin/env python3
"""Preflight checker for real-trajectory ingestion manifests (GitHub issue #3065).

Validates a bring-your-own-dataset real-trajectory ingestion manifest against the JSON Schema and
the fail-closed semantic contract (license acknowledgment, git-ignored staging, explicit durable
boundary, and benchmark eligibility gated on checksum-validated availability).

This tool reads only the manifest file. It downloads nothing, stages nothing, and never touches raw
external data.

Examples:
    uv run python scripts/tools/check_real_trajectory_manifest.py \
        configs/data/real_trajectory_manifest.example.yaml
    uv run python scripts/tools/check_real_trajectory_manifest.py my_manifest.yaml --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jsonschema

# Allow running as a plain script (python scripts/tools/...) without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robot_sf.data_ingestion.real_trajectory_contract import (
    ContractError,
    _resolved_staging_dir,
    _staging_tree_sha256,
    load_manifest,
    run_preflight,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Path to a YAML/JSON manifest file.")
    parser.add_argument(
        "--json", action="store_true", help="Emit a machine-readable JSON report instead of text."
    )
    parser.add_argument(
        "--staging-checksum",
        action="store_true",
        help=(
            "Include the resolved staging directory and aggregate SHA-256 tree checksum when "
            "staging files are locally available."
        ),
    )
    return parser


def _staging_tree_report(manifest: dict) -> dict[str, object]:
    staging = manifest.get("staging", {})
    staging_dir = staging.get("staging_dir") if isinstance(staging, dict) else None
    if not isinstance(staging_dir, str):
        return {"available": False, "reason": "manifest.staging.staging_dir missing"}

    resolved = _resolved_staging_dir(staging_dir)
    if "$" in str(resolved):
        return {
            "available": False,
            "staging_dir": str(resolved),
            "reason": "environment variable unresolved",
        }
    if not resolved.is_dir():
        return {
            "available": False,
            "staging_dir": str(resolved),
            "reason": "staging directory missing",
        }

    file_count = sum(1 for path in resolved.rglob("*") if path.is_file())
    if file_count == 0:
        return {
            "available": False,
            "staging_dir": str(resolved),
            "file_count": 0,
            "reason": "staging directory empty",
        }
    return {
        "available": True,
        "staging_dir": str(resolved),
        "file_count": file_count,
        "tree_sha256": _staging_tree_sha256(resolved),
    }


def main(argv: list[str] | None = None) -> int:
    """Validate one manifest and return a process exit code (0 ok, 1 contract violation, 2 input error)."""
    args = _build_parser().parse_args(argv)

    try:
        manifest = load_manifest(args.manifest)
        result = run_preflight(manifest)
    except (ContractError, jsonschema.ValidationError) as exc:
        if args.json:
            print(json.dumps({"manifest": str(args.manifest), "ok": False, "error": str(exc)}))
        else:
            print(f"FAIL: {args.manifest}: {exc}", file=sys.stderr)
        return 2

    if args.json:
        payload = {
            "manifest": str(args.manifest),
            "dataset_id": result.dataset_id,
            "availability": result.availability,
            "benchmark_eligibility": result.benchmark_eligibility,
            "ok": result.ok,
            "issues": [
                {"code": i.code, "severity": i.severity, "message": i.message}
                for i in result.issues
            ],
        }
        if args.staging_checksum:
            payload["staging_tree"] = _staging_tree_report(manifest)
        print(json.dumps(payload, indent=2))
    else:
        status = "OK" if result.ok else "FAIL"
        print(f"{status}: {args.manifest} (dataset_id={result.dataset_id})")
        print(
            f"  availability={result.availability} "
            f"benchmark_eligibility={result.benchmark_eligibility}"
        )
        if args.staging_checksum:
            print(f" staging_tree={_staging_tree_report(manifest)}")
        for issue in result.issues:
            print(f"  [{issue.severity}] {issue.code}: {issue.message}")

    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
