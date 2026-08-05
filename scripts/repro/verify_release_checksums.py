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
import re
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


def _resolve_frozen_source_commit(
    manifest: dict[str, Any],
    repo_root: Path,
) -> tuple[str | None, str | None]:
    """Validate the optional Git revision that freezes repository evidence."""

    raw_commit = manifest.get("frozen_manifest_origin_main_commit")
    if raw_commit is None:
        return None, None
    if not isinstance(raw_commit, str) or re.fullmatch(r"[0-9a-fA-F]{40}", raw_commit) is None:
        return None, "frozen_manifest_origin_main_commit must be a 40-character Git SHA."

    revision = raw_commit.lower()
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--verify", f"{revision}^{{commit}}"],
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        return None, f"Could not inspect frozen source commit {revision}: {exc}"
    if completed.returncode != 0:
        return None, f"Frozen source commit {revision} is unavailable in the repository."
    return revision, None


def _sha256_git_path(repo_root: Path, revision: str, relative_path: str) -> str | None:
    """Hash one repository path from an immutable Git tree."""

    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "cat-file", "blob", f"{revision}:{relative_path}"],
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return hashlib.sha256(completed.stdout).hexdigest()


def _verify_repository_entry(
    entry: Any,
    index: int,
    resolved_root: Path,
    repo_root: Path,
    source_commit: str | None,
) -> dict[str, Any]:
    """Verify one repository entry, preserving an explicitly frozen source hash."""
    if not isinstance(entry, dict):
        return {"match": False, "error": f"entries[{index}] must be a mapping."}

    raw_path = entry.get("path")
    expected_sha256 = entry.get("sha256")
    result: dict[str, Any] = {
        "path": raw_path,
        "expected_sha256": expected_sha256,
        "match": False,
    }
    if not isinstance(raw_path, str) or not raw_path:
        result["error"] = f"entries[{index}] is missing a non-empty path."
        return result
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in expected_sha256)
    ):
        result["error"] = f"entries[{index}] has an invalid SHA-256 digest."
        return result

    candidate = Path(raw_path)
    if candidate.is_absolute():
        result["error"] = "Repository entry path must be relative."
        return result
    path = (resolved_root / candidate).resolve()
    try:
        path.relative_to(resolved_root)
    except ValueError:
        result["error"] = "Repository entry path escapes the repository root."
        return result
    if not path.is_file():
        result["error"] = "Repository entry is missing or not a regular file."
        return result

    actual_sha256 = _sha256_file(path)
    result["actual_sha256"] = actual_sha256
    expected_sha256 = expected_sha256.lower()
    if actual_sha256 == expected_sha256:
        result["match"] = True
        return result

    if source_commit is not None:
        relative_path = path.relative_to(resolved_root).as_posix()
        frozen_sha256 = _sha256_git_path(repo_root, source_commit, relative_path)
        if frozen_sha256 == expected_sha256:
            result["current_sha256"] = actual_sha256
            result["actual_sha256"] = frozen_sha256
            result["source_commit"] = source_commit
            result["match"] = True
    return result


def _verify_repository_entries(
    entries: Any,
    repo_root: Path,
    *,
    source_commit: str | None = None,
) -> list[dict[str, Any]]:
    """Verify repository-relative checksum entries without a release archive.

    The current checkout must contain every entry. If a manifest explicitly records a
    frozen source commit, a current-byte mismatch is accepted only when the immutable
    Git blob at that commit exactly matches the manifest checksum.
    """
    if not isinstance(entries, list) or not entries:
        return [{"match": False, "error": "Manifest entries must be a non-empty list."}]

    resolved_root = repo_root.resolve()
    return [
        _verify_repository_entry(entry, index, resolved_root, repo_root, source_commit)
        for index, entry in enumerate(entries)
    ]


def _verify_manifest_repository_entries(
    manifest: dict[str, Any],
    repo_root: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Verify entries and return any frozen-source resolution error separately."""
    source_commit, source_commit_error = _resolve_frozen_source_commit(manifest, repo_root)
    if source_commit_error is not None:
        return [], [source_commit_error]
    return (
        _verify_repository_entries(
            manifest.get("entries"),
            repo_root,
            source_commit=source_commit,
        ),
        [],
    )


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


def _resolve_bundle_directory(
    bundle_evidence: dict[str, Any],
    repo_root: Path,
) -> tuple[str | None, Path | None, list[str]]:
    """Resolve and validate the optional directory-backed bundle root."""
    raw_directory = bundle_evidence.get("directory")
    if raw_directory is None:
        return None, None, []
    if not isinstance(raw_directory, str) or not raw_directory:
        return None, None, ["artifact_set.bundle_evidence.directory must be a non-empty path."]
    if Path(raw_directory).is_absolute():
        return (
            raw_directory,
            None,
            ["artifact_set.bundle_evidence.directory must be repository-relative."],
        )

    resolved_root = repo_root.resolve()
    directory = (resolved_root / raw_directory).resolve()
    try:
        directory.relative_to(resolved_root)
    except ValueError:
        return (
            raw_directory,
            None,
            ["artifact_set.bundle_evidence.directory escapes the repository root."],
        )
    if not directory.is_dir():
        return (
            raw_directory,
            None,
            ["artifact_set.bundle_evidence.directory is missing or not a directory."],
        )
    return raw_directory, directory, []


def _bundle_file_scope_error(
    path: str,
    directory: Path,
    bundle_directory: str,
    repo_root: Path,
) -> str | None:
    """Return an error when a declared bundle file is outside its directory."""
    resolved_path = (repo_root.resolve() / Path(path)).resolve()
    try:
        resolved_path.relative_to(directory)
    except ValueError:
        return (
            f"Bundle-evidence file {path!r} is outside the declared bundle directory "
            f"{bundle_directory!r}."
        )
    return None


def _verify_bundle_evidence_coverage(
    bundle_evidence: Any,
    entries: Any,
    repo_root: Path,
) -> list[str]:
    """Verify declared bundle files and, when configured, the whole bundle directory."""
    if not isinstance(bundle_evidence, dict):
        return ["artifact_set.bundle_evidence must be a mapping."]
    files = bundle_evidence.get("files")
    if not isinstance(files, list) or not files:
        return ["artifact_set.bundle_evidence.files must be a non-empty list."]

    bundle_directory, directory, directory_errors = _resolve_bundle_directory(
        bundle_evidence,
        repo_root,
    )

    entry_digests, errors = _repository_entry_digests(entries)
    bundle_paths: set[str] = set()
    for index, bundle_file in enumerate(files):
        errors.extend(
            _bundle_file_coverage_errors(index, bundle_file, entry_digests, bundle_paths),
        )

        if directory is not None and bundle_directory is not None and isinstance(bundle_file, dict):
            path = bundle_file.get("path")
            if isinstance(path, str) and path:
                scope_error = _bundle_file_scope_error(
                    path,
                    directory,
                    bundle_directory,
                    repo_root,
                )
                if scope_error is not None:
                    errors.append(scope_error)

    if directory is None:
        return [*errors, *directory_errors]

    resolved_root = repo_root.resolve()

    for path in sorted(candidate for candidate in directory.rglob("*") if candidate.is_file()):
        relative_path = path.relative_to(resolved_root).as_posix()
        if relative_path not in bundle_paths:
            errors.append(
                f"Bundle directory file {relative_path!r} is missing from checksum entries."
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
        resolved_repo_root = repo_root or Path.cwd()
        coverage_errors = _verify_bundle_evidence_coverage(
            artifact_set["bundle_evidence"],
            manifest.get("entries"),
            resolved_repo_root,
        )
        report["verdicts"]["bundle_evidence_coverage"] = {
            "match": not coverage_errors,
            "errors": coverage_errors,
        }
        report["errors"].extend(coverage_errors)
        repository_results, repository_errors = _verify_manifest_repository_entries(
            manifest,
            resolved_repo_root,
        )
        report["errors"].extend(repository_errors)
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
