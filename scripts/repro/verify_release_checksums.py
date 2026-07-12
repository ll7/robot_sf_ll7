#!/usr/bin/env python3
"""Verify release bundle checksums against the authoritative manifest.

Downloads (or uses a local copy of) the release bundle, verifies SHA-256
checksums against the manifest, and produces a structured verification report.

Usage:
    # Download and verify from GitHub release
    python scripts/repro/verify_release_checksums.py --tag 0.0.2

    # Verify a local bundle file
    python scripts/repro/verify_release_checksums.py --tag 0.0.2 --bundle-path /path/to/bundle.tar.gz

    # Verify with custom manifest
    python scripts/repro/verify_release_checksums.py --tag 0.0.2 --manifest configs/releases/release_0_0_2_checksum_manifest.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _env_info() -> dict[str, str]:
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "architecture": platform.machine(),
        "git_commit": _git_commit(),
    }


def _download_bundle(url: Path | str, dest: Path, tag: str) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    bundle_name = str(url).rsplit("/", 1)[-1]
    bundle_path = dest / bundle_name
    subprocess.check_call(
        ["gh", "release", "download", tag, "--pattern", bundle_name, "--dir", str(dest)],
    )
    return bundle_path


def _verify_bundle_checksum(bundle_path: Path, expected_sha256: str) -> dict[str, Any]:
    actual_sha256 = _sha256_file(bundle_path)
    return {
        "path": str(bundle_path),
        "expected_sha256": expected_sha256,
        "actual_sha256": actual_sha256,
        "match": actual_sha256 == expected_sha256,
        "size_bytes": bundle_path.stat().st_size,
    }


def _verify_embedded_artifacts(
    bundle_path: Path,
    embedded: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with tarfile.open(bundle_path, "r:gz") as tar:
        for name, info in embedded.items():
            archive_path = info["path_in_archive"]
            expected_sha = info["sha256"]
            result: dict[str, Any] = {
                "name": name,
                "archive_path": archive_path,
                "expected_sha256": expected_sha,
            }
            try:
                member = tar.getmember(archive_path)
                result["found"] = True
                result["size_bytes"] = member.size
                f = tar.extractfile(member)
                if f is not None:
                    actual_sha = hashlib.sha256(f.read()).hexdigest()
                    result["actual_sha256"] = actual_sha
                    result["match"] = actual_sha == expected_sha
                else:
                    result["actual_sha256"] = None
                    result["match"] = False
                    result["error"] = "Could not read file from archive"
            except KeyError:
                result["found"] = False
                result["match"] = False
                result["error"] = f"Path not found in archive: {archive_path}"
            results.append(result)
    return results


def _list_archive_contents(bundle_path: Path) -> list[str]:
    with tarfile.open(bundle_path, "r:gz") as tar:
        return sorted(tar.getnames())


def verify_release(  # noqa: C901 - each manifest/download/checksum failure needs a structured report.
    manifest_path: Path,
    bundle_path: Path | None,
    output_dir: Path,
    download: bool = True,
) -> dict[str, Any]:
    """Run the full release checksum verification.

    Returns:
        Structured verification report dict.
    """
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)

    if not isinstance(manifest, dict):
        return {
            "schema": "release-checksum-verification.v1",
            "created_at_utc": _utc_now_iso(),
            "manifest_path": str(manifest_path),
            "environment": _env_info(),
            "verdicts": {},
            "errors": ["Checksum manifest root must be a mapping."],
            "overall_verdict": "error",
        }

    release_tag = manifest.get("release_tag")
    if not isinstance(release_tag, str) or not release_tag:
        return {
            "schema": "release-checksum-verification.v1",
            "created_at_utc": _utc_now_iso(),
            "manifest_path": str(manifest_path),
            "environment": _env_info(),
            "verdicts": {},
            "errors": ["Checksum manifest must define a non-empty release_tag."],
            "overall_verdict": "error",
        }

    report: dict[str, Any] = {
        "schema": "release-checksum-verification.v1",
        "created_at_utc": _utc_now_iso(),
        "release_tag": release_tag,
        "release_id": manifest.get("release_id"),
        "manifest_path": str(manifest_path),
        "environment": _env_info(),
        "verdicts": {},
        "errors": [],
    }

    expected_bundle = manifest["artifact_set"]["bundle_archive"]
    bundle_sha256 = expected_bundle["sha256"]
    bundle_url = expected_bundle["url"]

    if bundle_path is None:
        if not download:
            report["errors"].append("No bundle path provided and download disabled.")
            report["overall_verdict"] = "error"
            return report
        try:
            bundle_path = _download_bundle(bundle_url, output_dir, release_tag)
            if not bundle_path.is_file():
                raise FileNotFoundError(f"Downloaded bundle file not found: {bundle_path}")
        except (subprocess.CalledProcessError, OSError) as exc:
            report["errors"].append(f"Bundle download failed: {exc}")
            report["overall_verdict"] = "error"
            return report

    report["bundle_path"] = str(bundle_path)
    report["verdicts"]["bundle_checksum"] = _verify_bundle_checksum(bundle_path, bundle_sha256)

    if not report["verdicts"]["bundle_checksum"]["match"]:
        report["errors"].append("Bundle checksum mismatch.")
        report["overall_verdict"] = "fail"
        return report

    embedded = manifest.get("embedded_artifacts", {})
    if embedded:
        report["verdicts"]["embedded_artifacts"] = _verify_embedded_artifacts(
            bundle_path,
            embedded,
        )
        for art in report["verdicts"]["embedded_artifacts"]:
            if not art.get("match"):
                report["errors"].append(
                    f"Embedded artifact {art['name']} checksum mismatch.",
                )

    report["archive_contents"] = _list_archive_contents(bundle_path)
    report["archive_file_count"] = len(report["archive_contents"])

    report["overall_verdict"] = "pass" if not report["errors"] else "fail"
    return report


def main() -> None:
    """Run release checksum verification from CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tag",
        default="0.0.2",
        help="Release tag to verify (default: 0.0.2)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Path to checksum manifest YAML",
    )
    parser.add_argument(
        "--bundle-path",
        type=Path,
        help="Local path to bundle tar.gz (skips download)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/release_verification"),
        help="Directory for downloaded bundle and report",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Disable automatic download from GitHub release",
    )
    args = parser.parse_args()

    manifest_path = args.manifest
    if manifest_path is None:
        tag_slug = args.tag.replace(".", "_")
        manifest_path = Path(f"configs/releases/release_{tag_slug}_checksum_manifest.yaml")
        if not manifest_path.exists():
            print(f"ERROR: No manifest found at {manifest_path}", file=sys.stderr)
            sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = verify_release(
        manifest_path=manifest_path,
        bundle_path=args.bundle_path,
        output_dir=args.output_dir,
        download=not args.no_download,
    )

    report_path = args.output_dir / "checksum_verification_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print(json.dumps(report, indent=2, sort_keys=True))

    if report["overall_verdict"] != "pass":
        print(f"\nVERDICT: {report['overall_verdict'].upper()}", file=sys.stderr)
        for err in report["errors"]:
            print(f"  ERROR: {err}", file=sys.stderr)
        sys.exit(1)

    print("\nVERDICT: PASS", file=sys.stderr)
    print(f"Report: {report_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
