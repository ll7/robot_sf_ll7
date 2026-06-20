#!/usr/bin/env python3
"""Build a dry-runable release evidence snapshot from tracked durable inputs."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

SCHEMA_VERSION = "release_evidence_snapshot.v0.1"
DEFAULT_INCLUDE_PATHS = (
    "CITATION.cff",
    "docs/benchmark_release_protocol.md",
    "docs/benchmark_release_reproducibility.md",
    "configs/benchmarks/releases/paper_experiment_matrix_v1_release_smoke_v0_1.yaml",
    "configs/benchmarks/paper_experiment_matrix_v1_release_smoke.yaml",
)
DEFAULT_CITATION_FILE = "CITATION.cff"
DEFAULT_RELEASE_TITLE = "Robot SF benchmark release evidence snapshot"
DEFAULT_REPOSITORY_URL = "https://github.com/ll7/robot_sf_ll7"
DEFAULT_DOI = "10.5281/zenodo.<record-id>"
DEFAULT_LICENSE = "GPL-3.0-or-later"
REFERENCE_ONLY_LINKS = {
    "artifact_backend": {
        "issue": 3075,
        "status": "reference_only",
        "note": (
            "Snapshot manifest keeps checksum and durable-reference fields compatible with a future "
            "artifact backend, but this command does not implement backend upload or hydration."
        ),
    },
    "researcher_guide": {
        "issue": 3071,
        "status": "reference_only",
        "note": (
            "Manifest fields are intended to be citable by a future researcher guide, but this "
            "command does not claim that guide is complete."
        ),
    },
    "campaign_result_store": {
        "issue": 3076,
        "status": "compatible_concept",
        "note": (
            "Result-store row-status and artifact-provenance concepts remain upstream inputs; this "
            "snapshot only records tracked files and catalog references."
        ),
    },
}
EXCLUSION_TOKENS = ("fallback", "degraded", "not_available", "unavailable", "failed")


class SnapshotError(RuntimeError):
    """Raised when the snapshot cannot be built because the repository is unreadable."""


def _build_parser() -> argparse.ArgumentParser:
    """Create the release evidence snapshot CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--source-ref",
        default="HEAD",
        help="Git ref to snapshot. Defaults to the current checkout HEAD.",
    )
    source.add_argument(
        "--tag",
        help="Git tag to snapshot. Equivalent to --source-ref <tag> and records tag metadata.",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "Additional tracked file or directory to include. May be repeated. Default release "
            "docs/config/citation inputs are always included unless --no-default-includes is set."
        ),
    )
    parser.add_argument(
        "--no-default-includes",
        action="store_true",
        help="Do not include the default release docs/config/citation files.",
    )
    parser.add_argument(
        "--artifact-catalog",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "Tracked artifact_catalog YAML/JSON to validate and include. May be repeated. "
            "By default, tracked catalogs under docs/context/evidence are auto-discovered."
        ),
    )
    parser.add_argument(
        "--no-auto-artifact-catalogs",
        action="store_true",
        help="Disable docs/context/evidence artifact catalog auto-discovery.",
    )
    parser.add_argument(
        "--require-input",
        action="append",
        default=[],
        metavar="PATH",
        help="Tracked file or directory that must exist at the source ref. Missing paths fail closed.",
    )
    parser.add_argument("--release-id", default=None, help="DOI-ready release identifier.")
    parser.add_argument(
        "--release-title",
        default=DEFAULT_RELEASE_TITLE,
        help="DOI-ready title to record in the manifest.",
    )
    parser.add_argument(
        "--repository-url",
        default=DEFAULT_REPOSITORY_URL,
        help="Repository URL to record in DOI-ready metadata.",
    )
    parser.add_argument("--doi", default=DEFAULT_DOI, help="DOI value or placeholder.")
    parser.add_argument("--license", default=DEFAULT_LICENSE, help="License identifier.")
    parser.add_argument(
        "--citation-file",
        default=DEFAULT_CITATION_FILE,
        help="Tracked citation file path.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the manifest JSON. Without this, the manifest is printed only.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the manifest and suppress --output-json writes.",
    )
    return parser


def _repo_root() -> Path:
    """Return the current Git repository root."""
    try:
        return Path(
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SnapshotError("not inside a Git repository") from exc


def _git_text(repo_root: Path, *args: str) -> str:
    """Run a Git command in ``repo_root`` and return stdout text."""
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=repo_root,
            text=True,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if isinstance(exc.stderr, str) else str(exc.stderr or "")
        raise SnapshotError(f"git {' '.join(args)} failed: {stderr}") from exc


def _git_bytes(repo_root: Path, *args: str) -> bytes:
    """Run a Git command in ``repo_root`` and return stdout bytes."""
    try:
        return subprocess.check_output(["git", *args], cwd=repo_root, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        stderr = (
            exc.stderr.decode("utf-8", errors="replace").strip()
            if isinstance(exc.stderr, bytes)
            else str(exc.stderr or "")
        )
        raise SnapshotError(f"git {' '.join(args)} failed: {stderr}") from exc


def _resolve_ref(repo_root: Path, source_ref: str) -> str:
    """Resolve a Git ref to a commit SHA."""
    return _git_text(repo_root, "rev-parse", f"{source_ref}^{{commit}}").strip()


def _tracked_path_exists(repo_root: Path, source_ref: str, path: str) -> bool:
    """Return whether a path exists in the Git tree for ``source_ref``."""
    normalized = _normalize_repo_path(path)
    if not normalized:
        return False
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{source_ref}:{normalized}"],
        cwd=repo_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode == 0:
        return True
    listing = _git_text(repo_root, "ls-tree", "--name-only", source_ref, "--", normalized)
    return bool(listing.strip())


def _list_tracked_files(repo_root: Path, source_ref: str, path: str) -> list[str]:
    """List tracked files at a file or directory path in stable order."""
    normalized = _normalize_repo_path(path)
    if not normalized:
        return []
    file_check = subprocess.run(
        ["git", "cat-file", "-e", f"{source_ref}:{normalized}"],
        cwd=repo_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if file_check.returncode == 0:
        return [normalized]
    listing = _git_text(repo_root, "ls-tree", "-r", "--name-only", source_ref, "--", normalized)
    return sorted(line.strip() for line in listing.splitlines() if line.strip())


def _read_blob(repo_root: Path, source_ref: str, path: str) -> bytes:
    """Read a tracked file blob from a Git ref."""
    return _git_bytes(repo_root, "show", f"{source_ref}:{_normalize_repo_path(path)}")


def _file_entry(repo_root: Path, source_ref: str, path: str, role: str) -> dict[str, Any]:
    """Build one file-manifest entry with a SHA-256 over Git blob bytes."""
    blob = _read_blob(repo_root, source_ref, path)
    return {
        "path": _normalize_repo_path(path),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "size_bytes": len(blob),
        "role": role,
        "source": f"git:{source_ref}",
    }


def _load_mapping_from_ref(repo_root: Path, source_ref: str, path: str) -> dict[str, Any]:
    """Load a YAML or JSON mapping from a tracked file at ``source_ref``."""
    raw = _read_blob(repo_root, source_ref, path).decode("utf-8")
    try:
        payload = json.loads(raw) if Path(path).suffix.lower() == ".json" else yaml.safe_load(raw)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise SnapshotError(f"could not parse {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SnapshotError(f"expected mapping payload in {path}")
    return payload


def _discover_artifact_catalogs(repo_root: Path, source_ref: str) -> list[str]:
    """Return tracked artifact catalog paths under docs/context/evidence."""
    listing = _git_text(
        repo_root,
        "ls-tree",
        "-r",
        "--name-only",
        source_ref,
        "--",
        "docs/context/evidence",
    )
    return sorted(
        path
        for path in (line.strip() for line in listing.splitlines())
        if Path(path).name
        in {"artifact_catalog.yaml", "artifact_catalog.yml", "artifact_catalog.json"}
    )


def _analyze_artifact_catalog(  # noqa: C901
    repo_root: Path,
    source_ref: str,
    catalog_path: str,
) -> tuple[dict[str, Any], list[str], list[dict[str, str]]]:
    """Validate one artifact catalog and return catalog summary, file paths, and exclusions."""
    catalog = _load_mapping_from_ref(repo_root, source_ref, catalog_path)
    catalog_dir = str(Path(catalog_path).parent)
    issues: list[str] = []
    referenced_paths = [catalog_path]
    exclusions: list[dict[str, str]] = []
    principal_artifacts: list[dict[str, Any]] = []

    artifacts = catalog.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        issues.append("catalog has no artifacts")
        artifacts = []

    for artifact in artifacts:
        if not isinstance(artifact, dict):
            issues.append("catalog artifact entry is not a mapping")
            continue
        refs = _artifact_file_refs(artifact)
        output_refs = _output_refs(artifact.get("outputs"))
        for ref in [*refs, *output_refs]:
            resolved = _resolve_catalog_ref(catalog_dir, ref.get("path"))
            expected = str(ref.get("sha256", "")).strip()
            if not resolved:
                issues.append("catalog file reference is missing path")
                continue
            referenced_paths.append(resolved)
            if not _tracked_path_exists(repo_root, source_ref, resolved):
                issues.append(f"referenced file is missing: {resolved}")
                continue
            actual = _file_entry(repo_root, source_ref, resolved, "artifact_catalog_reference")[
                "sha256"
            ]
            if expected and actual != expected:
                issues.append(
                    f"checksum mismatch for {resolved}: expected {expected}, got {actual}"
                )
            elif not expected:
                issues.append(f"referenced file is missing checksum: {resolved}")

        claim_boundary = str(artifact.get("claim_boundary", ""))
        token_hits = [token for token in EXCLUSION_TOKENS if token in claim_boundary.lower()]
        if token_hits:
            exclusions.append(
                {
                    "artifact_id": str(artifact.get("artifact_id", "unknown")),
                    "status_tokens": ",".join(token_hits),
                    "reason": "claim boundary excludes fallback/degraded/unavailable/failed rows",
                    "catalog": catalog_path,
                }
            )
        if artifact.get("artifact_kind") in {"table", "figure"}:
            principal_artifacts.append(
                {
                    "artifact_id": str(artifact.get("artifact_id", "")),
                    "artifact_kind": str(artifact.get("artifact_kind", "")),
                    "source_kind": str(artifact.get("source_kind", "")),
                    "outputs": sorted(ref["path"] for ref in output_refs if ref.get("path")),
                    "claim_boundary": claim_boundary,
                }
            )

    summary = {
        "path": catalog_path,
        "status": "valid" if not issues else "invalid",
        "catalog_id": str(catalog.get("catalog_id", "")),
        "artifact_count": len(artifacts),
        "principal_artifacts": principal_artifacts,
        "issues": issues,
    }
    return summary, sorted(set(referenced_paths)), exclusions


def _artifact_file_refs(artifact: Mapping[str, Any]) -> list[dict[str, str]]:
    """Return source and caption file refs from one artifact catalog entry."""
    refs: list[dict[str, str]] = []
    source_files = artifact.get("source_files")
    if isinstance(source_files, list):
        refs.extend(ref for ref in source_files if isinstance(ref, dict))
    caption = artifact.get("caption_file")
    if isinstance(caption, dict):
        refs.append(caption)
    return refs


def _output_refs(outputs: Any) -> list[dict[str, str]]:
    """Return output file refs from one artifact catalog entry."""
    if not isinstance(outputs, dict):
        return []
    return [ref for ref in outputs.values() if isinstance(ref, dict)]


def _resolve_catalog_ref(catalog_dir: str, raw_path: Any) -> str:
    """Resolve an artifact-catalog file reference to a repository-relative path."""
    if not isinstance(raw_path, str) or not raw_path.strip():
        return ""
    path = Path(raw_path.strip())
    if path.is_absolute() or ".." in path.parts:
        return ""
    candidate = _normalize_repo_path(str(Path(catalog_dir) / path))
    return candidate


def _extract_config_seed_identifiers(
    repo_root: Path,
    source_ref: str,
    paths: Iterable[str],
) -> dict[str, Any]:
    """Extract lightweight config, release, and seed identifiers from included mappings."""
    configs: list[dict[str, Any]] = []
    seed_policies: list[dict[str, Any]] = []
    releases: list[dict[str, Any]] = []
    for path in sorted(set(paths)):
        suffix = Path(path).suffix.lower()
        if suffix not in {".yaml", ".yml", ".json"}:
            continue
        try:
            payload = _load_mapping_from_ref(repo_root, source_ref, path)
        except Exception:
            continue
        if path.startswith("configs/"):
            configs.append(
                {
                    "path": path,
                    "name": payload.get("name"),
                    "paper_facing": payload.get("paper_facing"),
                    "release_tag": payload.get("release_tag"),
                    "scenario_matrix": payload.get("scenario_matrix")
                    or (payload.get("scenario") or {}).get("matrix_path")
                    if isinstance(payload.get("scenario"), dict)
                    else payload.get("scenario_matrix"),
                }
            )
        seed_policy = payload.get("seed_policy")
        if isinstance(seed_policy, dict):
            seed_policies.append(
                {
                    "path": path,
                    "mode": seed_policy.get("mode"),
                    "seed_set": seed_policy.get("seed_set"),
                    "seeds": seed_policy.get("seeds"),
                    "seed_sets_path": seed_policy.get("seed_sets_path"),
                }
            )
        if payload.get("release_id") or payload.get("release_tag"):
            releases.append(
                {
                    "path": path,
                    "release_id": payload.get("release_id"),
                    "release_tag": payload.get("release_tag"),
                    "benchmark_protocol_version": payload.get("benchmark_protocol_version"),
                }
            )
    return {
        "configs": configs,
        "seed_policies": seed_policies,
        "release_manifests": releases,
    }


def _metadata_from_args(args: argparse.Namespace, identifiers: Mapping[str, Any]) -> dict[str, Any]:
    """Build DOI-ready metadata, filling release_id from included release manifests when possible."""
    release_id = args.release_id
    if release_id is None:
        releases = identifiers.get("release_manifests")
        if isinstance(releases, list) and releases:
            for candidate in releases:
                if not isinstance(candidate, dict):
                    continue
                candidate_id = str(candidate.get("release_id") or "").strip()
                if candidate_id:
                    release_id = candidate_id
                    break
    return {
        "release_id": release_id or "unassigned",
        "title": args.release_title,
        "doi": args.doi,
        "repository_url": args.repository_url,
        "license": args.license,
        "citation_file": args.citation_file,
        "metadata_status": "doi_ready_fields_present",
    }


def _dirty_worktree(repo_root: Path) -> bool:
    """Return whether tracked files are dirty in the current checkout."""
    return bool(_git_text(repo_root, "status", "--porcelain", "--untracked-files=no").strip())


def _normalize_repo_path(path: str) -> str:
    """Normalize a path for Git tree lookups."""
    return Path(path).as_posix().strip("/")


def build_snapshot(  # noqa: C901
    args: argparse.Namespace, *, repo_root: Path | None = None
) -> dict[str, Any]:
    """Build the release evidence snapshot payload."""
    root = repo_root or _repo_root()
    source_ref = args.tag or args.source_ref
    source_commit = _resolve_ref(root, source_ref)
    include_paths = [] if args.no_default_includes else list(DEFAULT_INCLUDE_PATHS)
    include_paths.extend(args.include)

    required_inputs: list[dict[str, str]] = []
    missing_inputs: list[dict[str, str]] = []
    file_roles: dict[str, set[str]] = {}
    artifact_catalogs: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []

    for path in args.require_input:
        normalized = _normalize_repo_path(path)
        if _tracked_path_exists(root, source_ref, normalized):
            required_inputs.append({"path": normalized, "status": "present"})
            for file_path in _list_tracked_files(root, source_ref, normalized):
                file_roles.setdefault(file_path, set()).add("required_input")
        else:
            record = {
                "path": normalized,
                "status": "missing",
                "reason": "required input is absent from the source ref",
            }
            required_inputs.append(record)
            missing_inputs.append(record)

    for path in include_paths:
        normalized = _normalize_repo_path(path)
        if not _tracked_path_exists(root, source_ref, normalized):
            missing_inputs.append(
                {
                    "path": normalized,
                    "status": "missing",
                    "reason": "included input is absent from the source ref",
                }
            )
            continue
        for file_path in _list_tracked_files(root, source_ref, normalized):
            file_roles.setdefault(file_path, set()).add("included_input")

    catalog_paths = [_normalize_repo_path(path) for path in args.artifact_catalog]
    if not args.no_auto_artifact_catalogs:
        catalog_paths.extend(_discover_artifact_catalogs(root, source_ref))
    catalog_paths = sorted(set(catalog_paths))
    if not catalog_paths:
        missing_inputs.append(
            {
                "path": "docs/context/evidence/**/artifact_catalog.yaml",
                "status": "missing",
                "reason": "no tracked artifact catalog found for a principal promoted table/figure",
            }
        )
    for catalog_path in catalog_paths:
        if not _tracked_path_exists(root, source_ref, catalog_path):
            missing_inputs.append(
                {
                    "path": catalog_path,
                    "status": "missing",
                    "reason": "artifact catalog is absent from the source ref",
                }
            )
            continue
        try:
            summary, referenced_paths, catalog_exclusions = _analyze_artifact_catalog(
                root, source_ref, catalog_path
            )
        except SnapshotError as exc:
            summary = {
                "path": catalog_path,
                "status": "invalid",
                "catalog_id": "",
                "artifact_count": 0,
                "principal_artifacts": [],
                "issues": [str(exc)],
            }
            referenced_paths = [catalog_path]
            catalog_exclusions = []
        artifact_catalogs.append(summary)
        exclusions.extend(catalog_exclusions)
        for file_path in referenced_paths:
            file_roles.setdefault(file_path, set()).add("artifact_catalog_reference")

    file_manifest = [
        _file_entry(root, source_ref, path, ",".join(sorted(roles)))
        for path, roles in sorted(file_roles.items())
        if _tracked_path_exists(root, source_ref, path)
    ]
    identifiers = _extract_config_seed_identifiers(root, source_ref, file_roles)
    catalog_invalid = any(catalog["status"] != "valid" for catalog in artifact_catalogs)
    status = "fail_closed" if missing_inputs or catalog_invalid else "valid"
    evidence_classification = (
        "diagnostic-only"
        if status != "valid" or _catalogs_are_diagnostic_only(artifact_catalogs)
        else "release-candidate"
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "evidence_classification": evidence_classification,
        "source": {
            "source_ref": source_ref,
            "tag": args.tag,
            "source_commit": source_commit,
            "worktree_dirty": _dirty_worktree(root) if source_ref == "HEAD" else None,
            "generated_at_utc": dt.datetime.now(dt.timezone.utc)  # noqa: UP017
            .isoformat()
            .replace("+00:00", "Z"),
            "command": shlex.join(
                [sys.executable, str(Path(__file__)), *getattr(args, "invoked_args", [])]
            ),
        },
        "doi_ready_metadata": _metadata_from_args(args, identifiers),
        "required_inputs": required_inputs,
        "missing_inputs": missing_inputs,
        "file_manifest": {
            "file_count": len(file_manifest),
            "total_bytes": sum(int(entry["size_bytes"]) for entry in file_manifest),
            "files": file_manifest,
        },
        "config_seed_identifiers": identifiers,
        "artifact_catalogs": artifact_catalogs,
        "fallback_degraded_exclusions": exclusions,
        "conceptual_links": REFERENCE_ONLY_LINKS,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the release evidence snapshot CLI."""
    raw_argv = list(argv) if argv is not None else list(sys.argv[1:])
    args = _build_parser().parse_args(raw_argv)
    args.invoked_args = raw_argv
    try:
        payload = build_snapshot(args)
    except SnapshotError as exc:
        sys.stderr.write(f"{Path(__file__).name}: error: {exc}\n")
        return 2

    output = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None and not args.dry_run:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(output, encoding="utf-8")
    sys.stdout.write(output)
    return 0 if payload["status"] == "valid" else 2


def _catalogs_are_diagnostic_only(catalogs: Iterable[Mapping[str, Any]]) -> bool:
    """Return whether principal artifacts explicitly limit themselves to diagnostic use."""
    for catalog in catalogs:
        artifacts = catalog.get("principal_artifacts")
        if not isinstance(artifacts, list):
            continue
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                continue
            boundary = str(artifact.get("claim_boundary", "")).lower()
            if "diagnostic-only" in boundary or "not standalone benchmark evidence" in boundary:
                return True
    return False


if __name__ == "__main__":
    raise SystemExit(main())
