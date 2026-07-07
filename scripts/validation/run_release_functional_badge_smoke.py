#!/usr/bin/env python3
"""Run release functional-badge smoke checks on an artifact bundle.

Extracts/inspects the bundle, verifies all checksums, and executes a
smoke check on the bundle's content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    """Compute the SHA-256 hash of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_bundle_checksums(bundle_dir: Path, files_list: list[dict[str, Any]]) -> list[str]:
    """Verify checksums of files in payload directory.

    Returns:
        List of validation error messages.
    """
    errors = []
    payload_dir = bundle_dir / "payload"
    for entry in files_list:
        rel_path = entry.get("path")
        expected_sha = entry.get("sha256")
        if not rel_path or not expected_sha:
            errors.append(f"Invalid file entry in manifest: {entry}")
            continue

        file_path = payload_dir / rel_path
        if not file_path.exists():
            errors.append(f"Missing file in payload: {rel_path}")
            continue

        actual_sha = sha256_file(file_path)
        if actual_sha != expected_sha:
            errors.append(
                f"Checksum mismatch for {rel_path}: expected {expected_sha}, got {actual_sha}"
            )
    return errors


def _check_summaries(payload_dir: Path, errors: list[str]) -> None:
    """Smoke check: Load and parse campaign_summary or summary.json if present."""
    summary_files = list(payload_dir.glob("**/campaign_summary.json")) + list(
        payload_dir.glob("**/summary.json")
    )
    for sf in summary_files:
        try:
            data = json.loads(sf.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                errors.append(f"Summary file {sf.name} is not a JSON object.")
        except (json.JSONDecodeError, OSError) as e:
            errors.append(f"Failed to read/parse summary file {sf.name}: {e}")


def _check_episodes(payload_dir: Path, errors: list[str]) -> None:
    """Smoke check: Check episodes.jsonl is readable and has lines."""
    episodes_files = list(payload_dir.glob("**/episodes.jsonl"))
    for ef in episodes_files:
        try:
            with ef.open("r", encoding="utf-8") as handle:
                lines = handle.readlines()
                if not lines:
                    errors.append(f"Episodes file {ef.name} is empty.")
                for line in lines[:5]:  # smoke check first few lines
                    json.loads(line)
        except (json.JSONDecodeError, OSError) as e:
            errors.append(f"Failed to parse episodes file {ef.name} lines: {e}")


def _check_report_re_derivation(payload_dir: Path, errors: list[str]) -> None:
    """Functional check: Re-derive headline table and byte-compare it against shipped report."""
    campaign_summary_files = list(payload_dir.glob("**/campaign_summary.json"))
    campaign_report_files = list(payload_dir.glob("**/campaign_report.md"))

    if campaign_summary_files and campaign_report_files:
        summary_path = campaign_summary_files[0]
        report_path = campaign_report_files[0]
        try:
            import tempfile

            from robot_sf.benchmark.camera_ready._reporting import (
                write_campaign_report,
            )

            payload_data = json.loads(summary_path.read_text(encoding="utf-8"))
            with tempfile.TemporaryDirectory() as tmpdir:
                temp_report_path = Path(tmpdir) / "campaign_report.md"
                write_campaign_report(temp_report_path, payload_data)

                # Byte-compare after normalizing line endings
                shipped_content = report_path.read_text(encoding="utf-8").replace("\r\n", "\n")
                derived_content = temp_report_path.read_text(encoding="utf-8").replace("\r\n", "\n")

                if shipped_content != derived_content:
                    errors.append(
                        "Byte-comparison failed: re-derived headline table does not match shipped campaign_report.md."
                    )
        except (OSError, ValueError, json.JSONDecodeError) as e:
            errors.append(f"Failed to re-derive headline table and byte-compare: {e}")


def execute_functional_smoke_checks(bundle_dir: Path) -> list[str]:
    """Execute smoke checks on the bundle artifacts (e.g.

    parsing metadata/episodes/summaries to ensure they are readable).

    Returns:
        List of check failure messages.
    """
    errors = []
    # Find manifest
    manifest_paths = list(bundle_dir.glob("*manifest.json"))
    if not manifest_paths:
        errors.append("No manifest.json found in the bundle root.")
        return errors

    manifest_path = manifest_paths[0]
    try:
        json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        errors.append(f"Failed to parse manifest JSON: {e}")
        return errors

    payload_dir = bundle_dir / "payload"
    _check_summaries(payload_dir, errors)
    _check_episodes(payload_dir, errors)
    _check_report_re_derivation(payload_dir, errors)

    return errors


def main(argv: list[str] | None = None) -> int:
    """Run release functional-badge smoke checks from command line."""
    parser = argparse.ArgumentParser(description="Run release functional-badge smoke campaign.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--bundle-path", type=Path, help="Directory of the unpacked release bundle.")
    group.add_argument(
        "--bundle-archive", type=Path, help="Tarball (.tar.gz) of the release bundle."
    )
    parser.add_argument("--output", type=Path, default=Path("functional_badge_report.json"))
    args = parser.parse_args(argv)

    temp_dir = None
    bundle_dir = None

    try:
        if args.bundle_archive:
            if not args.bundle_archive.exists():
                print(f"Error: Archive does not exist: {args.bundle_archive}", file=sys.stderr)
                return 1
            temp_dir = tempfile.TemporaryDirectory()
            print(f"Extracting {args.bundle_archive} to temporary directory...")
            with tarfile.open(args.bundle_archive, "r:gz") as tar:
                tar.extractall(path=temp_dir.name)
            # Find the top level directory inside the archive
            extracted_paths = list(Path(temp_dir.name).iterdir())
            if len(extracted_paths) == 1 and extracted_paths[0].is_dir():
                bundle_dir = extracted_paths[0]
            else:
                bundle_dir = Path(temp_dir.name)
        else:
            bundle_dir = args.bundle_path
            if not bundle_dir.exists() or not bundle_dir.is_dir():
                print(
                    f"Error: Bundle path does not exist or is not a directory: {bundle_dir}",
                    file=sys.stderr,
                )
                return 1

        print(f"Running smoke verification on bundle: {bundle_dir}")

        # Find publication/evidence manifest
        manifest_paths = list(bundle_dir.glob("*_manifest.json"))
        if not manifest_paths:
            print(
                "Error: No manifest file ending in _manifest.json found in bundle.", file=sys.stderr
            )
            return 1

        manifest_path = manifest_paths[0]
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"Error parsing manifest JSON: {e}", file=sys.stderr)
            return 1

        files_list = manifest.get("files", [])

        # Verify all checksums
        checksum_errors = verify_bundle_checksums(bundle_dir, files_list)

        # Execute smoke checks
        smoke_errors = execute_functional_smoke_checks(bundle_dir)

        all_errors = checksum_errors + smoke_errors
        status = "passed" if not all_errors else "failed"

        report = {
            "schema_version": "functional-badge-smoke-report.v1",
            "timestamp_utc": _utc_now_iso(),
            "status": status,
            "checksum_verified": len(checksum_errors) == 0,
            "checks_verified": len(smoke_errors) == 0,
            "errors": all_errors,
        }

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Functional smoke report written to: {args.output}")

        if status == "passed":
            print("SUCCESS: Functional smoke test passed.")
            return 0
        else:
            print(
                f"FAILURE: Functional smoke test failed with {len(all_errors)} errors:",
                file=sys.stderr,
            )
            for err in all_errors:
                print(f" - {err}", file=sys.stderr)
            return 1

    finally:
        if temp_dir:
            temp_dir.cleanup()


if __name__ == "__main__":
    sys.exit(main())
