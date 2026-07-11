#!/usr/bin/env python3
"""Report traceability gaps in committed evidence-registry campaign records.

The default mode is intentionally non-blocking so existing provenance gaps can be
measured before this checker becomes a CI gate. Use ``--strict`` to make findings
fail the command.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from collections.abc import Sequence


COMMIT_RE = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{40}(?![0-9a-fA-F])")
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
MARKDOWN_CAMPAIGN_RE = re.compile(r"\bcampaign_id\s*[:=]\s*`?([A-Za-z0-9_.-]+)`?")
CONFIG_PATH_KEYS = {
    "config",
    "config_path",
    "producing_config",
    "source_config",
    "training_config",
}
ARTIFACT_PATH_KEYS = {
    "artifact_path",
    "artifact_uri",
    "file",
    "filename",
    "path",
    "source_path",
}


def _issue(path: Path, code: str, message: str) -> dict[str, str]:
    """Return one stable, machine-readable lint finding."""
    return {"path": path.as_posix(), "code": code, "message": message}


def _git_succeeds(repo_root: Path, *args: str) -> bool:
    """Return whether a Git query succeeds without exposing Git diagnostics."""
    return (
        subprocess.run(
            ["git", *args],
            cwd=repo_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )


def _git_bytes(repo_root: Path, *args: str) -> bytes | None:
    """Return Git command output bytes, or ``None`` when the object is unavailable."""

    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout if result.returncode == 0 else None


def _config_hash_matches(
    repo_root: Path, commit: str, path: str, declared_hashes: set[str]
) -> bool:
    """Return whether one committed config blob matches a declared SHA-256 value."""

    blob = _git_bytes(repo_root, "show", f"{commit}:{path}")
    return blob is not None and hashlib.sha256(blob).hexdigest() in declared_hashes


def _is_tracked(repo_root: Path, repo_path: str) -> bool:
    """Return whether a normalized repository path is tracked in the current index."""
    return _git_succeeds(repo_root, "ls-files", "--error-unmatch", "--", repo_path)


def _resolve_repo_path(repo_root: Path, value: str) -> tuple[str, Path] | None:
    """Normalize a repository-relative artifact path, rejecting URLs and escapes."""
    if "://" in value or value.startswith("urn:"):
        return None
    candidate = (repo_root / value).resolve()
    try:
        normalized = candidate.relative_to(repo_root.resolve())
    except ValueError:
        return None
    return normalized.as_posix(), candidate


def _load_document(path: Path) -> Any:
    """Load supported structured registry files, returning text for Markdown/CSV."""
    raw = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(raw)
    if suffix in {".yaml", ".yml"}:
        return yaml.safe_load(raw)
    if suffix == ".csv":
        return list(csv.DictReader(raw.splitlines()))
    return raw


def _iter_mappings(
    value: Any, ancestors: tuple[Mapping[str, Any], ...] = ()
) -> Iterable[tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]]]:
    """Yield mappings with their parent mappings for inherited path/location metadata."""
    if isinstance(value, Mapping):
        yield value, ancestors
        for child in value.values():
            yield from _iter_mappings(child, (*ancestors, value))
    elif isinstance(value, list):
        for child in value:
            yield from _iter_mappings(child, ancestors)


def _string_values(mapping: Mapping[str, Any], keys: set[str]) -> list[str]:
    """Return non-empty strings from case-insensitive field names."""
    return [
        value.strip()
        for key, value in mapping.items()
        if isinstance(key, str) and key.lower() in keys and isinstance(value, str) and value.strip()
    ]


def _campaign_ids(value: Any) -> list[str]:
    """Extract explicit campaign identifiers without treating README vocabulary as data."""
    if isinstance(value, str):
        return MARKDOWN_CAMPAIGN_RE.findall(value)
    values: list[str] = []
    for mapping, _ancestors in _iter_mappings(value):
        campaign_id = mapping.get("campaign_id")
        if isinstance(campaign_id, str) and campaign_id.strip():
            values.append(campaign_id.strip())
    return values


def _file_metadata(value: Any) -> tuple[list[str], list[str], list[str]]:
    """Return config paths, config checksums, and full commit references in a file."""
    if isinstance(value, str):
        return [], [], COMMIT_RE.findall(value)
    config_paths: list[str] = []
    config_hashes: list[str] = []
    commits: list[str] = []
    for mapping, _ancestors in _iter_mappings(value):
        config_paths.extend(_string_values(mapping, CONFIG_PATH_KEYS))
        for key, item in mapping.items():
            if not isinstance(key, str):
                continue
            lowered = key.lower()
            if lowered == "config_sha256" and isinstance(item, str):
                config_hashes.append(item.strip())
            if isinstance(item, str):
                commits.extend(COMMIT_RE.findall(item))
    return config_paths, config_hashes, commits


def _artifact_path(
    mapping: Mapping[str, Any], ancestors: tuple[Mapping[str, Any], ...]
) -> str | None:
    """Find a neighboring artifact path, including ``reports_dir`` filename manifests."""
    paths = _string_values(mapping, ARTIFACT_PATH_KEYS)
    for value in paths:
        if value and not value.startswith("configs/"):
            return value
    filename = next(
        (
            key
            for parent in reversed(ancestors)
            for key, child in parent.items()
            if child is mapping and isinstance(key, str) and Path(key).suffix
        ),
        None,
    )
    if filename is not None:
        reports_dir = next(
            (
                value
                for parent in reversed(ancestors)
                for value in _string_values(parent, {"reports_dir", "artifact_dir", "directory"})
            ),
            None,
        )
        return str(Path(reports_dir) / filename) if reports_dir else filename
    return None


def _has_location(mapping: Mapping[str, Any], ancestors: tuple[Mapping[str, Any], ...]) -> bool:
    """Return whether an uncommitted artifact has an explicit location marker."""
    return any(
        isinstance(value, str) and value.strip()
        for candidate in (*ancestors, mapping)
        for key, value in candidate.items()
        if isinstance(key, str) and (key.lower() == "location" or key.lower().endswith("_location"))
    )


def _artifact_hash_finding(
    repo_root: Path,
    display_path: Path,
    key: str,
    declared_hash: str,
    mapping: Mapping[str, Any],
    ancestors: tuple[Mapping[str, Any], ...],
) -> dict[str, str] | None:
    """Check one artifact hash declaration, returning its finding when invalid."""
    if not SHA256_RE.fullmatch(declared_hash):
        return _issue(display_path, "invalid_sha256", f"{key} is not a 64-hex SHA-256")
    artifact_path = _artifact_path(mapping, ancestors)
    if artifact_path is None:
        return _issue(
            display_path, "hash_without_artifact_path", f"{key} lacks an adjacent artifact path"
        )
    resolved = _resolve_repo_path(repo_root, artifact_path)
    if resolved is None or not _is_tracked(repo_root, resolved[0]):
        if not _has_location(mapping, ancestors):
            return _issue(
                display_path,
                "uncommitted_artifact_missing_location",
                f"{artifact_path} is not tracked and lacks an explicit location marker",
            )
        return None
    actual_hash = hashlib.sha256(resolved[1].read_bytes()).hexdigest()
    if actual_hash != declared_hash.lower():
        return _issue(
            display_path,
            "artifact_hash_mismatch",
            f"{artifact_path} SHA-256 does not match the tracked file",
        )
    return None


def _artifact_findings(repo_root: Path, display_path: Path, value: Any) -> list[dict[str, str]]:
    """Validate artifact checksum declarations against tracked artifact paths."""
    if isinstance(value, str):
        return []
    findings: list[dict[str, str]] = []
    for mapping, ancestors in _iter_mappings(value):
        for key, declared_hash in mapping.items():
            if not isinstance(key, str) or key.lower() not in {"sha256", "source_sha256"}:
                continue
            if not isinstance(declared_hash, str):
                continue
            finding = _artifact_hash_finding(
                repo_root, display_path, key, declared_hash, mapping, ancestors
            )
            if finding:
                findings.append(finding)
    return findings


def _campaign_metadata_findings(  # noqa: C901 - rule-to-finding mapping is intentionally linear
    repo_root: Path,
    display_path: Path,
    campaign_ids: Sequence[str],
    config_paths: Sequence[str],
    config_hashes: Sequence[str],
    commits: Sequence[str],
) -> list[dict[str, str]]:
    """Validate campaign-required identifiers and config existence at declared commits."""
    findings: list[dict[str, str]] = []
    if not campaign_ids:
        return findings
    if not config_paths:
        findings.append(
            _issue(
                display_path, "missing_config_path", "campaign_id requires a producing config path"
            )
        )
    if not config_hashes:
        findings.append(
            _issue(display_path, "missing_config_sha256", "campaign_id requires config_sha256")
        )
    elif any(not SHA256_RE.fullmatch(item) for item in config_hashes):
        findings.append(
            _issue(display_path, "invalid_config_sha256", "config_sha256 is not a 64-hex SHA-256")
        )
    if not commits:
        findings.append(
            _issue(display_path, "missing_commit", "campaign_id requires a full producing commit")
        )
    resolved_commits = [
        commit
        for commit in set(commits)
        if _git_succeeds(repo_root, "cat-file", "-e", f"{commit}^{{commit}}")
    ]
    declared_config_hashes = {item.lower() for item in config_hashes if SHA256_RE.fullmatch(item)}
    for config_path in sorted(set(config_paths)):
        normalized = _resolve_repo_path(repo_root, config_path)
        if normalized is None:
            findings.append(
                _issue(
                    display_path,
                    "invalid_config_path",
                    f"config path {config_path!r} is not repository-relative",
                )
            )
        elif not any(
            _git_succeeds(repo_root, "cat-file", "-e", f"{commit}:{normalized[0]}")
            for commit in resolved_commits
        ):
            findings.append(
                _issue(
                    display_path,
                    "config_missing_at_commit",
                    f"config {normalized[0]} does not exist at a declared commit",
                )
            )
        elif declared_config_hashes:
            config_hash_matches = any(
                _config_hash_matches(repo_root, commit, normalized[0], declared_config_hashes)
                for commit in resolved_commits
            )
            if not config_hash_matches:
                findings.append(
                    _issue(
                        display_path,
                        "config_sha256_mismatch",
                        f"config_sha256 does not match {normalized[0]} at a declared commit",
                    )
                )
    return findings


def _lint_document(repo_root: Path, path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Lint one registry file and return its campaign IDs plus findings."""
    display_path = path.relative_to(repo_root)
    try:
        value = _load_document(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, yaml.YAMLError) as exc:
        return [], [_issue(display_path, "unreadable_document", str(exc))]
    campaign_ids = _campaign_ids(value)
    config_paths, config_hashes, commits = _file_metadata(value)
    findings: list[dict[str, str]] = []
    for commit in sorted(set(commits)):
        if not _git_succeeds(repo_root, "cat-file", "-e", f"{commit}^{{commit}}"):
            findings.append(
                _issue(display_path, "dangling_commit", f"commit {commit} does not resolve")
            )
    findings.extend(
        _campaign_metadata_findings(
            repo_root, display_path, campaign_ids, config_paths, config_hashes, commits
        )
    )
    findings.extend(_artifact_findings(repo_root, display_path, value))
    return campaign_ids, findings


def lint_evidence_registry(repo_root: Path, registry_root: Path) -> dict[str, Any]:
    """Return a deterministic integrity report for every supported evidence file."""
    repo_root = repo_root.resolve()
    registry_root = registry_root.resolve()
    campaigns: dict[str, list[Path]] = defaultdict(list)
    findings: list[dict[str, str]] = []
    files = sorted(
        path
        for path in registry_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".json", ".yaml", ".yml", ".md", ".csv"}
    )
    for path in files:
        campaign_ids, document_findings = _lint_document(repo_root, path)
        findings.extend(document_findings)
        for campaign_id in campaign_ids:
            campaigns[campaign_id].append(path.relative_to(repo_root))
    for campaign_id, paths in sorted(campaigns.items()):
        if len(paths) > 1:
            for path in paths:
                findings.append(
                    _issue(
                        path,
                        "duplicate_campaign_id",
                        f"campaign_id {campaign_id!r} appears in {len(paths)} files",
                    )
                )
    findings.sort(key=lambda item: (item["path"], item["code"], item["message"]))
    by_code = dict(sorted(Counter(item["code"] for item in findings).items()))
    return {
        "registry_root": registry_root.relative_to(repo_root).as_posix(),
        "checked_files": len(files),
        "campaign_ids": sorted(campaigns),
        "issues": findings,
        "summary": {"findings": len(findings), "by_code": by_code},
    }


def _repo_root_from_git() -> Path:
    """Return the current checkout root for the default command invocation."""
    return Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    )


def main(argv: list[str] | None = None) -> int:
    """Run the report-only or strict evidence registry linter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=None, help="Repository root (defaults to Git root)."
    )
    parser.add_argument(
        "--registry-root",
        type=Path,
        default=Path("docs/context/evidence"),
        help="Evidence registry root, relative to --repo-root by default.",
    )
    parser.add_argument(
        "--strict", action="store_true", help="Exit nonzero when findings are present."
    )
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve() if args.repo_root else _repo_root_from_git()
    registry_root = (
        args.registry_root if args.registry_root.is_absolute() else repo_root / args.registry_root
    )
    report = lint_evidence_registry(repo_root, registry_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if args.strict and report["issues"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
