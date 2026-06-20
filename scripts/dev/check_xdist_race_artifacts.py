#!/usr/bin/env python3
"""Check xdist stress runs for shared-output race artifacts."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "xdist_race_artifact_scan.v1"
DEFAULT_SCAN_PATHS = (Path("output/tmp"), Path("output/coverage"))
TEMP_SUFFIXES = (".lock", ".part", ".partial", ".pid", ".tmp")
WORKER_PATTERN = re.compile(r"(^|[^A-Za-z0-9])gw\d+([^A-Za-z0-9]|$)")


@dataclass(frozen=True)
class ArtifactRecord:
    """Stable metadata for one scanned file."""

    path: str
    size: int


@dataclass(frozen=True)
class ArtifactViolation:
    """A generated artifact that suggests xdist shared-output races."""

    code: str
    path: str
    detail: str


def _git(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run a git command in *cwd*."""
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)


def _repo_root(cwd: Path) -> Path:
    """Return the enclosing git worktree root."""
    result = _git(["rev-parse", "--show-toplevel"], cwd=cwd)
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError("check_xdist_race_artifacts.py requires a git worktree")
    return Path(result.stdout.strip()).resolve()


def _as_repo_relative(path: Path, *, repo_root: Path) -> str:
    """Return *path* relative to *repo_root* using POSIX separators."""
    return path.resolve().relative_to(repo_root).as_posix()


def _resolve_scan_path(path: Path, *, repo_root: Path) -> Path:
    """Resolve and validate one scan path."""
    resolved = (repo_root / path).resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"scan path is outside repository: {path}") from exc
    return resolved


def _collect_records(paths: list[Path], *, repo_root: Path) -> dict[str, ArtifactRecord]:
    """Collect file records under the requested shared-output paths."""
    records: dict[str, ArtifactRecord] = {}
    for raw_path in paths:
        root = _resolve_scan_path(raw_path, repo_root=repo_root)
        if not root.exists():
            continue
        if root.is_file():
            rel = _as_repo_relative(root, repo_root=repo_root)
            records[rel] = ArtifactRecord(path=rel, size=root.stat().st_size)
            continue
        for child in sorted(item for item in root.rglob("*") if item.is_file()):
            rel = _as_repo_relative(child, repo_root=repo_root)
            records[rel] = ArtifactRecord(path=rel, size=child.stat().st_size)
    return records


def _load_baseline(path: Path | None) -> set[str]:
    """Load a prior baseline manifest."""
    if path is None or not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(record["path"]) for record in payload.get("records", [])}


def _write_manifest(path: Path, *, records: dict[str, ArtifactRecord], repo_root: Path) -> None:
    """Write a baseline or scan manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SCHEMA_VERSION,
        "repo_root": str(repo_root),
        "records": [asdict(record) for record in records.values()],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _is_under(path: str, prefix: str) -> bool:
    """Return whether a repo-relative path is under a repo-relative prefix."""
    normalized = prefix.strip("/")
    return path == normalized or path.startswith(f"{normalized}/")


def _default_allowed_prefixes(run_id: str) -> list[str]:
    """Return output prefixes expected from the xdist race route."""
    return [
        f"output/tmp/xdist-race/{run_id}",
        "output/coverage",
    ]


def _classify_violations(
    *,
    records: dict[str, ArtifactRecord],
    baseline_paths: set[str],
    run_id: str,
    allowed_prefixes: list[str],
    scan_paths: list[Path],
) -> list[ArtifactViolation]:
    """Classify new suspicious files produced during the stress run."""
    isolated_prefix = f"output/tmp/xdist-race/{run_id}"
    scanned_prefixes = [path.as_posix() for path in scan_paths]
    violations: list[ArtifactViolation] = []
    for path, record in sorted(records.items()):
        is_new = path not in baseline_paths
        if not is_new:
            continue
        if record.size == 0:
            violations.append(
                ArtifactViolation(
                    code="truncated_zero_byte",
                    path=path,
                    detail="new generated file is zero bytes",
                )
            )
        if path.endswith(TEMP_SUFFIXES):
            violations.append(
                ArtifactViolation(
                    code="orphaned_temp_file",
                    path=path,
                    detail=f"new generated file has a temporary suffix {TEMP_SUFFIXES}",
                )
            )
        if WORKER_PATTERN.search(Path(path).name) and not _is_under(path, isolated_prefix):
            violations.append(
                ArtifactViolation(
                    code="cross_worker_artifact",
                    path=path,
                    detail="worker-tagged file escaped the isolated xdist race artifact root",
                )
            )
        if not any(_is_under(path, prefix) for prefix in allowed_prefixes) and not any(
            _is_under(path, scan_prefix) for scan_prefix in scanned_prefixes
        ):
            violations.append(
                ArtifactViolation(
                    code="orphaned_shared_output",
                    path=path,
                    detail="new generated file is outside allowed xdist race output prefixes",
                )
            )
    return violations


def scan_xdist_race_artifacts(
    *,
    paths: list[Path],
    run_id: str,
    baseline_path: Path | None = None,
    output_manifest: Path | None = None,
    allowed_prefixes: list[str] | None = None,
    cwd: Path | None = None,
) -> dict[str, Any]:
    """Scan shared-output paths and return a compact JSON-compatible result."""
    repo_root = _repo_root(cwd or Path.cwd())
    records = _collect_records(paths, repo_root=repo_root)
    if output_manifest is not None:
        _write_manifest(output_manifest, records=records, repo_root=repo_root)
    baseline_paths = _load_baseline(baseline_path)
    effective_prefixes = allowed_prefixes or _default_allowed_prefixes(run_id)
    violations = _classify_violations(
        records=records,
        baseline_paths=baseline_paths,
        run_id=run_id,
        allowed_prefixes=effective_prefixes,
        scan_paths=paths,
    )
    return {
        "schema": SCHEMA_VERSION,
        "status": "passed" if not violations else "failed",
        "run_id": run_id,
        "scan_paths": [path.as_posix() for path in paths],
        "allowed_prefixes": effective_prefixes,
        "baseline_path": str(baseline_path) if baseline_path else None,
        "output_manifest": str(output_manifest) if output_manifest else None,
        "scanned_file_count": len(records),
        "new_file_count": len(set(records) - baseline_paths),
        "violations": [asdict(violation) for violation in violations],
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        default="local",
        help="Stable run id used to identify the isolated xdist race artifact root.",
    )
    parser.add_argument(
        "--baseline-json",
        type=Path,
        default=None,
        help="Baseline manifest written before the stress run.",
    )
    parser.add_argument(
        "--write-manifest",
        type=Path,
        default=None,
        help="Write the current scan manifest to this path.",
    )
    parser.add_argument(
        "--allowed-prefix",
        action="append",
        default=[],
        help="Repo-relative prefix allowed for new files; may be repeated.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full scan payload as JSON.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=list(DEFAULT_SCAN_PATHS),
        help="Repo-relative shared-output paths to scan.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    try:
        result = scan_xdist_race_artifacts(
            paths=args.paths,
            run_id=args.run_id,
            baseline_path=args.baseline_json,
            output_manifest=args.write_manifest,
            allowed_prefixes=args.allowed_prefix or None,
        )
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"xdist_race_artifact_scan={result['status']}")
        print(f"scanned_file_count={result['scanned_file_count']}")
        print(f"new_file_count={result['new_file_count']}")
        for violation in result["violations"]:
            print(f"{violation['code']}: {violation['path']} ({violation['detail']})")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
