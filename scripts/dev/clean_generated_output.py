#!/usr/bin/env python3
"""Safely remove generated output paths while preserving tracked files."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _git(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run a git command in *cwd*."""
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def _git_z(args: list[str], *, cwd: Path) -> list[Path]:
    """Return NUL-delimited path output from a git command."""
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    return [
        cwd / raw.decode("utf-8", errors="surrogateescape")
        for raw in result.stdout.split(b"\0")
        if raw
    ]


def _repo_root(cwd: Path) -> Path | None:
    """Return the enclosing git worktree root, if available."""
    result = _git(["rev-parse", "--show-toplevel"], cwd=cwd)
    if result.returncode != 0:
        return None
    return Path(result.stdout.strip()).resolve()


def _is_within(path: Path, root: Path) -> bool:
    """Return whether *path* is inside *root*."""
    try:
        path.resolve().relative_to(root)
    except ValueError:
        return False
    return True


def _remove_path(path: Path) -> None:
    """Remove a generated file, symlink, or directory."""
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _contains_tracked_children(path: Path, repo_root: Path) -> bool:
    """Return whether *path* is a directory with tracked children."""
    if not path.is_dir() or path.is_symlink():
        return False
    rel = path.relative_to(repo_root).as_posix()
    return bool(_git(["ls-files", "--", rel], cwd=repo_root).stdout.strip())


def _remove_empty_dirs(root: Path, repo_root: Path) -> None:
    """Remove empty untracked directories left after generated files are removed."""
    if not root.exists() or not root.is_dir():
        return
    for dirpath, _, _ in os.walk(root, topdown=False):
        path = Path(dirpath)
        if path == repo_root:
            continue
        try:
            if any(path.iterdir()):
                continue
        except OSError:
            continue
        rel = path.relative_to(repo_root).as_posix()
        tracked = _git(["ls-files", "--error-unmatch", rel], cwd=repo_root)
        if tracked.returncode == 0:
            continue
        try:
            path.rmdir()
        except OSError:
            pass


def clean_paths(paths: list[Path], *, cwd: Path) -> int:
    """Clean generated files under *paths* while preserving tracked files."""
    repo_root = _repo_root(cwd)
    if repo_root is None:
        print("clean_generated_output requires a git worktree", file=sys.stderr)
        return 2

    cleaned = 0
    for raw_path in paths:
        path = (cwd / raw_path).resolve()
        if not _is_within(path, repo_root):
            print(f"refusing to clean path outside repository: {raw_path}", file=sys.stderr)
            return 2
        if not path.exists():
            continue

        rel = path.relative_to(repo_root).as_posix()
        tracked_root = _git(["ls-files", "--error-unmatch", rel], cwd=repo_root).returncode == 0
        tracked_children = _git(["ls-files", "--", rel], cwd=repo_root).stdout.splitlines()
        if not tracked_root and not tracked_children:
            result = _git(["clean", "-fdx", "--", rel], cwd=repo_root)
            if result.returncode != 0:
                print(result.stderr.strip() or f"git clean failed for {rel}", file=sys.stderr)
                return result.returncode
            cleaned += 1
            continue

        generated = {
            *(_git_z(["ls-files", "-z", "-o", "--exclude-standard", "--", rel], cwd=repo_root)),
            *(
                _git_z(
                    ["ls-files", "-z", "-o", "-i", "--exclude-standard", "--", rel], cwd=repo_root
                )
            ),
        }
        for target in sorted(generated, key=lambda item: len(item.parts), reverse=True):
            if target.exists() and _is_within(target, repo_root):
                if _contains_tracked_children(target, repo_root):
                    continue
                _remove_path(target)
                cleaned += 1
        _remove_empty_dirs(path, repo_root)

    print(f"cleaned_generated_paths={cleaned}")
    return 0


def eligibility_gate(record_path: Path, artifact_id: str) -> int:
    """Check one artifact identity before deleting anything.

    Check-only integration for the shared cleanup-eligibility guard (#8906): it never
    deletes, prints the public-safe report, and returns 0 only when ``eligible``.
    """
    from scripts.validation.check_cleanup_eligibility import (
        RecordError,
        check_artifact_file,
        render_text,
    )

    try:
        report = check_artifact_file(record_path, artifact_id)
    except RecordError as exc:
        print(f"cleanup-eligibility: {exc}", file=sys.stderr)
        return 2
    print(render_text(report))
    return 0 if report.outcome == "eligible" else 2


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Generated output paths to clean without deleting tracked files.",
    )
    parser.add_argument(
        "--eligibility-record",
        type=Path,
        help="Cleanup-eligibility record to gate deletion on (check-only).",
    )
    parser.add_argument(
        "--artifact",
        help="Artifact identity to evaluate in the eligibility record.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Evaluate eligibility only; never delete, even when eligible.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    if args.eligibility_record is not None or args.artifact is not None:
        if args.eligibility_record is None or args.artifact is None:
            print("--eligibility-record and --artifact must be used together", file=sys.stderr)
            return 2
        code = eligibility_gate(args.eligibility_record, args.artifact)
        if code != 0 or args.check:
            return code
    if not args.paths:
        print("no paths to clean and no eligibility check requested", file=sys.stderr)
        return 2
    return clean_paths(args.paths, cwd=Path.cwd())


if __name__ == "__main__":
    raise SystemExit(main())
