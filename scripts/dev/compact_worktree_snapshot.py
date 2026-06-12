#!/usr/bin/env python3
"""Emit a compact worktree bootstrap-state snapshot.

This helper detects fresh linked worktrees and reports bootstrap state for autonomous
PR loops. It avoids broad git worktree output by reporting only essential fields.

Use before running expensive commands in a linked worktree to determine whether
bootstrap steps (uv sync, local.machine.md symlink) are required.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

SCHEMA_VERSION = "compact_worktree_snapshot.v1"


@dataclass(frozen=True, slots=True)
class WorktreeInfo:
    """Compact worktree state."""

    path: str
    branch: str
    head_sha: str
    is_current: bool
    is_linked: bool
    is_fresh: bool
    has_local_machine: bool
    has_venv: bool
    main_repo_root: str | None
    bootstrap_required: bool


@dataclass(frozen=True, slots=True)
class SnapshotResult:
    """Full snapshot result."""

    schema: str
    current_worktree: str | None
    current_branch: str
    current_head_sha: str
    is_linked_worktree: bool
    filters: list[str]
    worktrees: list[WorktreeInfo]
    worktree_count: int
    included_worktree_count: int
    worktrees_truncated: bool
    errors: list[str]


def _run_command(
    args: list[str],
    *,
    cwd: str | None = None,
    timeout: int = 30,
) -> subprocess.CompletedProcess:
    """Run a command and capture output."""
    try:
        return subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            cwd=cwd,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            args=args,
            returncode=124,
            stdout="",
            stderr=f"command timed out after {timeout} seconds",
        )


def _git_common_dir() -> Path | None:
    """Return the common git dir path."""
    result = _run_command(["git", "rev-parse", "--git-common-dir"])
    if result.returncode != 0:
        return None
    return Path(result.stdout.strip())


def _is_linked_worktree() -> bool:
    """Check if current checkout is a linked worktree."""
    git_dir = Path(".git")
    if not git_dir.exists():
        return False
    if git_dir.is_file():
        content = git_dir.read_text().strip()
        return content.startswith("gitdir:")
    return False


def _get_main_repo_root() -> Path | None:
    """Get the main repository root from common git dir."""
    common_dir = _git_common_dir()
    if not common_dir:
        return None
    parent = common_dir.parent
    if parent.exists() and (parent / ".git").exists():
        return parent
    return None


def _parse_worktree_porcelain(stdout: str) -> list[dict[str, str]]:  # noqa: C901
    """Parse git worktree list --porcelain output."""
    worktrees: list[dict[str, str]] = []
    current: dict[str, str] = {}

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            if current:
                worktrees.append(current)
                current = {}
            continue

        parts = line.split(" ", 1)
        if len(parts) != 2:
            continue

        key, value = parts
        if key == "worktree":
            if current:
                worktrees.append(current)
            current = {"path": value}
        elif key == "HEAD":
            current["head"] = value
        elif key == "branch":
            current["branch"] = value.removeprefix("refs/heads/")
        elif key in {"bare", "detached"}:
            current[key] = "true"

    if current:
        worktrees.append(current)

    return worktrees


def _is_fresh_worktree(worktree_path: Path) -> bool:
    """Check if a worktree needs bootstrap."""
    has_local_machine = (worktree_path / "local.machine.md").exists()
    has_venv = (worktree_path / ".venv").exists()
    return not has_local_machine and not has_venv


def _matches_filters(row: dict[str, str], filters: list[str]) -> bool:
    """Return whether a worktree row matches any issue/branch/slug filter."""
    if not filters:
        return True
    haystack = " ".join((row.get("path", ""), row.get("branch", ""))).lower()
    return any(filter_text.lower() in haystack for filter_text in filters)


def _detect_worktrees(
    *,
    current_path: Path,
    main_repo_root: Path | None,
    filters: list[str],
    limit: int | None = 20,
) -> tuple[list[WorktreeInfo], int, bool]:
    """Detect all worktrees with bootstrap state."""
    result = _run_command(["git", "worktree", "list", "--porcelain"])
    if result.returncode != 0:
        return [], 0, False

    parsed = _parse_worktree_porcelain(result.stdout)
    filtered = [row for row in parsed if _matches_filters(row, filters)]
    selected = filtered if limit is None else filtered[:limit]
    worktrees = []

    for wt in selected:
        wt_path = Path(wt.get("path", ""))
        is_current = wt_path.resolve() == current_path.resolve()
        is_linked = _is_linked_worktree() if is_current else wt_path.joinpath(".git").is_file()
        has_local_machine = wt_path.joinpath("local.machine.md").exists()
        has_venv = wt_path.joinpath(".venv").exists()
        is_fresh = _is_fresh_worktree(wt_path)

        worktrees.append(
            WorktreeInfo(
                path=str(wt_path),
                branch=wt.get("branch", ""),
                head_sha=wt.get("head", ""),
                is_current=is_current,
                is_linked=is_linked,
                is_fresh=is_fresh,
                has_local_machine=has_local_machine,
                has_venv=has_venv,
                main_repo_root=str(main_repo_root) if main_repo_root else None,
                bootstrap_required=is_fresh and main_repo_root is not None,
            )
        )

    return worktrees, len(parsed), len(filtered) > len(selected)


def build_snapshot(
    *,
    include_all_worktrees: bool = False,
    worktree_limit: int = 20,
    filters: list[str] | None = None,
) -> SnapshotResult:
    """Build the compact worktree snapshot."""
    errors: list[str] = []
    filter_values = filters or []
    current_path = Path.cwd().resolve()

    result = _run_command(["git", "branch", "--show-current"])
    if result.returncode != 0:
        errors.append("failed to get current branch")
        current_branch = ""
    else:
        current_branch = result.stdout.strip()

    result = _run_command(["git", "rev-parse", "HEAD"])
    if result.returncode != 0:
        errors.append("failed to get HEAD SHA")
        current_head_sha = ""
    else:
        current_head_sha = result.stdout.strip()

    is_linked = _is_linked_worktree()
    main_root = _get_main_repo_root()

    worktrees, total_worktree_count, worktrees_truncated = _detect_worktrees(
        current_path=current_path,
        main_repo_root=main_root,
        filters=filter_values,
        limit=None if include_all_worktrees else worktree_limit,
    )

    current_worktree = None
    for wt in worktrees:
        if wt.is_current:
            current_worktree = wt.path
            break

    return SnapshotResult(
        schema=SCHEMA_VERSION,
        current_worktree=current_worktree,
        current_branch=current_branch,
        current_head_sha=current_head_sha,
        is_linked_worktree=is_linked,
        filters=filter_values,
        worktrees=worktrees,
        worktree_count=total_worktree_count,
        included_worktree_count=len(worktrees),
        worktrees_truncated=worktrees_truncated,
        errors=errors,
    )


def format_human(result: SnapshotResult) -> str:
    """Format snapshot as human-readable text."""
    lines = [
        f"Worktree Snapshot (schema: {result.schema})",
        f"  Current: {result.current_worktree or 'N/A'}",
        f"  Branch: {result.current_branch or 'N/A'}",
        f"  HEAD: {result.current_head_sha or 'N/A'}",
        f"  Linked worktree: {result.is_linked_worktree}",
        f"  Total worktrees: {result.worktree_count}",
        f"  Included worktrees: {result.included_worktree_count}",
        f"  Truncated: {result.worktrees_truncated}",
    ]
    if result.filters:
        lines.append(f"  Filters: {', '.join(result.filters)}")

    if result.worktrees:
        lines.append("  Worktrees:")
        for wt in result.worktrees:
            status = []
            if wt.is_current:
                status.append("current")
            if wt.is_fresh:
                status.append("fresh")
            if wt.bootstrap_required:
                status.append("bootstrap-required")
            status_str = f" ({', '.join(status)})" if status else ""
            lines.append(f"    - {wt.branch}: {wt.path}{status_str}")

    if result.errors:
        lines.append("  Errors:")
        for err in result.errors:
            lines.append(f"    - {err}")

    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON",
    )
    parser.add_argument(
        "--include-all-worktrees",
        action="store_true",
        help="Include all worktrees without limit",
    )
    parser.add_argument(
        "--worktree-limit",
        type=int,
        default=20,
        help="Maximum worktrees to include",
    )
    parser.add_argument(
        "--filter",
        dest="filters",
        action="append",
        default=[],
        help="Issue number, branch substring, or slug to match against branch/path. May repeat.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)

    try:
        result = build_snapshot(
            include_all_worktrees=args.include_all_worktrees,
            worktree_limit=args.worktree_limit,
            filters=args.filters,
        )
    except Exception as exc:
        print(f"ERROR building worktree snapshot: {exc}", file=sys.stderr)
        return 1

    if args.json:
        output = asdict(result)
        print(json.dumps(output, indent=2, sort_keys=True))
    else:
        print(format_human(result))

    return 0 if not result.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
