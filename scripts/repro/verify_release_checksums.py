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


def _verify_repository_entries(entries: Any, repo_root: Path) -> list[dict[str, Any]]:
    """Verify repository-relative checksum entries without a release archive."""
    if not isinstance(entries, list) or not entries:
        return [{"match": False, "error": "Manifest entries must be a non-empty list."}]

    resolved_root = repo_root.resolve()
    results: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            results.append({"match": False, "error": f"entries[{index}] must be a mapping."})
            continue
        raw_path = entry.get("path")
        expected_sha256 = entry.get("sha256")
        result: dict[str, Any] = {
            "path": raw_path,
            "expected_sha256": expected_sha256,
            "match": False,
        }
        if not isinstance(raw_path, str) or not raw_path:
            result["error"] = f"entries[{index}] is missing a non-empty path."
            results.append(result)
            continue
        if (
            not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in expected_sha256)
        ):
            result["error"] = f"entries[{index}] has an invalid SHA-256 digest."
            results.append(result)
            continue

        candidate = Path(raw_path)
        if candidate.is_absolute():
            result["error"] = "Repository entry path must be relative."
            results.append(result)
            continue
        path = (resolved_root / candidate).resolve()
        try:
            path.relative_to(resolved_root)
        except ValueError:
            result["error"] = "Repository entry path escapes the repository root."
            results.append(result)
            continue
        if not path.is_file():
            result["error"] = "Repository entry is missing or not a regular file."
            results.append(result)
            continue

        actual_sha256 = _sha256_file(path)
        result["actual_sha256"] = actual_sha256
        result["match"] = actual_sha256 == expected_sha256.lower()
        results.append(result)
    return results


def _repository_entry_digests(entries: Any) -> tuple[dict[str, str], list[str]]:
    """Index repository checksum entries by path for bundle-coverage validation."""
    if not isinstance(entries, list) or not entries:
        return {}, ["Manifest entries must be a non-empty list."]

    entry_digests: dict[str, str] = {}
    errors: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        path = entry.get("path")
        digest = entry.get("sha256")
        if not isinstance(path, str) or not path or not isinstance(digest, str):
            continue
        if path in entry_digests:
            errors.append(f"Duplicate repository checksum entry for {path!r}.")
            continue
        entry_digests[path] = digest.lower()
    return entry_digests, errors


def _bundle_file_coverage_errors(
    index: int,
    bundle_file: Any,
    entry_digests: dict[str, str],
    bundle_paths: set[str],
) -> list[str]:
    """Validate one bundle file declaration against repository checksum entries."""
    if not isinstance(bundle_file, dict):
        return [f"artifact_set.bundle_evidence.files[{index}] must be a mapping."]
    path = bundle_file.get("path")
    digest = bundle_file.get("sha256")
    if not isinstance(path, str) or not path:
        return [f"artifact_set.bundle_evidence.files[{index}] is missing a non-empty path."]
    if path in bundle_paths:
        return [f"Duplicate bundle-evidence file declaration for {path!r}."]
    bundle_paths.add(path)
    if not isinstance(digest, str) or not digest:
        return [f"Bundle-evidence file {path!r} is missing a SHA-256 digest."]
    entry_digest = entry_digests.get(path)
    if entry_digest is None:
        return [f"Bundle-evidence file {path!r} is missing from checksum entries."]
    if entry_digest != digest.lower():
        return [f"Bundle-evidence file {path!r} has a checksum inconsistent with entries."]
    return []


def _verify_bundle_evidence_coverage(bundle_evidence: Any, entries: Any) -> list[str]:
    """Verify every declared bundle-evidence file has an identical checksum entry."""
    if not isinstance(bundle_evidence, dict):
        return ["artifact_set.bundle_evidence must be a mapping."]
    files = bundle_evidence.get("files")
    if not isinstance(files, list) or not files:
        return ["artifact_set.bundle_evidence.files must be a non-empty list."]

    entry_digests, errors = _repository_entry_digests(entries)
    bundle_paths: set[str] = set()
    for index, bundle_file in enumerate(files):
        errors.extend(
            _bundle_file_coverage_errors(index, bundle_file, entry_digests, bundle_paths),
        )
    return errors


def verify_release(  # noqa: C901, PLR0912 - failures need distinct structured reports.
    manifest_path: Path,
    bundle_path: Path | None,
    output_dir: Path,
    download: bool = True,
    repo_root: Path | None = None,
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

    artifact_set = manifest.get("artifact_set")
    if not isinstance(artifact_set, dict):
        report["errors"].append("Checksum manifest must define an artifact_set mapping.")
        report["overall_verdict"] = "error"
        return report

    has_bundle_archive = "bundle_archive" in artifact_set
    has_bundle_evidence = "bundle_evidence" in artifact_set
    if has_bundle_archive and has_bundle_evidence:
        report["errors"].append(
            "artifact_set must define exactly one of bundle_archive or bundle_evidence.",
        )
        report["overall_verdict"] = "error"
        return report
    if has_bundle_evidence:
        coverage_errors = _verify_bundle_evidence_coverage(
            artifact_set["bundle_evidence"],
            manifest.get("entries"),
        )
        report["verdicts"]["bundle_evidence_coverage"] = {
            "match": not coverage_errors,
            "errors": coverage_errors,
        }
        report["errors"].extend(coverage_errors)
        repository_results = _verify_repository_entries(
            manifest.get("entries"),
            repo_root or Path.cwd(),
        )
        report["verdicts"]["repository_entries"] = repository_results
        for result in repository_results:
            if not result.get("match"):
                report["errors"].append(
                    f"Repository entry checksum verification failed: {result.get('path', 'unknown')}",
                )
        report["overall_verdict"] = "pass" if not report["errors"] else "fail"
        return report
    if not has_bundle_archive:
        report["errors"].append(
            "artifact_set must define exactly one of bundle_archive or bundle_evidence.",
        )
        report["overall_verdict"] = "error"
        return report
    expected_bundle = artifact_set["bundle_archive"]
    if not isinstance(expected_bundle, dict):
        report["errors"].append("artifact_set.bundle_archive must be a mapping.")
        report["overall_verdict"] = "error"
        return report
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
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root for manifests that verify tracked entries instead of an archive.",
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
        repo_root=args.repo_root,
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
