#!/usr/bin/env python3
"""Emit a compact worktree hygiene snapshot.

This helper is read-only. It summarizes branch drift, dirty worktrees, detached
heads, and missing upstreams without printing full `git worktree` output. Use it
before remote maintenance, stale-worktree cleanup planning, or broad PR loops.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCHEMA_VERSION = "worktree_hygiene_snapshot.v1"


@dataclass(frozen=True, slots=True)
class WorktreeHygiene:
    """A compact per-worktree hygiene row."""

    path: str
    branch: str
    head_sha: str
    is_current: bool
    is_detached: bool
    dirty_entries: int
    upstream: str | None
    ahead: int | None
    behind: int | None
    issues: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class RepoStatus:
    """Optional status for the current checkout."""

    branch_status: str
    dirty_entries: int
    ahead: int | None
    behind: int | None


@dataclass(frozen=True, slots=True)
class HygieneSnapshot:
    """Full worktree hygiene snapshot."""

    schema: str
    current_worktree: str | None
    total_worktrees: int
    included_worktrees: int
    worktrees_truncated: bool
    filters: list[str]
    issue_counts: dict[str, int]
    repo_status: RepoStatus | None
    worktrees: list[WorktreeHygiene]
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
    except OSError as exc:
        return subprocess.CompletedProcess(
            args=args,
            returncode=127,
            stdout="",
            stderr=str(exc),
        )


def _parse_worktree_porcelain(stdout: str) -> list[dict[str, str]]:
    """Parse `git worktree list --porcelain` rows."""
    worktrees: list[dict[str, str]] = []
    current: dict[str, str] = {}

    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                worktrees.append(current)
                current = {}
            continue
        parts = line.split(" ", 1)
        if len(parts) != 2:
            if line == "detached":
                current["detached"] = "true"
            continue
        key, value = parts
        if key == "worktree":
            if current:
                worktrees.append(current)
            current = {"path": value}
        elif key == "HEAD":
            current["head_sha"] = value
        elif key == "branch":
            current["branch"] = value.removeprefix("refs/heads/")
        elif key == "detached":
            current["detached"] = "true"

    if current:
        worktrees.append(current)

    return worktrees


def _matches_filters(row: dict[str, str], filters: list[str]) -> bool:
    """Return whether a worktree row matches any branch/path filter."""
    if not filters:
        return True
    haystack = " ".join((row.get("path", ""), row.get("branch", ""))).lower()
    return any(value.lower() in haystack for value in filters)


def _dirty_entry_count(path: str) -> int:
    """Count short-status rows for a worktree."""
    result = _run_command(["git", "status", "--porcelain"], cwd=path)
    if result.returncode != 0:
        return -1
    return len([line for line in result.stdout.splitlines() if line.strip()])


def _upstream(path: str) -> str | None:
    """Return the configured upstream branch, if any."""
    result = _run_command(["git", "rev-parse", "--abbrev-ref", "@{upstream}"], cwd=path)
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def _ahead_behind(path: str, upstream: str | None) -> tuple[int | None, int | None]:
    """Return ahead and behind counts relative to upstream."""
    if not upstream:
        return None, None
    result = _run_command(
        ["git", "rev-list", "--left-right", "--count", f"HEAD...{upstream}"], cwd=path
    )
    if result.returncode != 0:
        return None, None
    parts = result.stdout.split()
    if len(parts) != 2:
        return None, None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None, None


def _classify_issues(
    *,
    branch: str,
    is_detached: bool,
    dirty_entries: int,
    upstream: str | None,
    ahead: int | None,
    behind: int | None,
) -> list[str]:
    """Classify hygiene issues for a worktree."""
    issues: list[str] = []
    if is_detached:
        issues.append("detached")
    if dirty_entries < 0:
        issues.append("status_failed")
    elif dirty_entries > 0:
        issues.append("dirty")
    if branch and not upstream:
        issues.append("missing_upstream")
    if ahead:
        issues.append("ahead")
    if behind:
        issues.append("behind")
    return issues


def _repo_status() -> RepoStatus | None:
    """Build optional status for the current checkout."""
    status = _run_command(["git", "status", "--short", "--branch"])
    if status.returncode != 0:
        return None
    lines = status.stdout.splitlines()
    branch_status = lines[0] if lines else ""
    upstream = _upstream(".")
    ahead, behind = _ahead_behind(".", upstream)
    return RepoStatus(
        branch_status=branch_status,
        dirty_entries=max(0, len(lines) - 1),
        ahead=ahead,
        behind=behind,
    )


def _build_row(row: dict[str, str], current_path: Path) -> WorktreeHygiene:
    """Build one hygiene row from a parsed worktree row."""
    path = row.get("path", "")
    branch = row.get("branch", "")
    is_detached = row.get("detached") == "true" or not branch
    dirty_entries = _dirty_entry_count(path)
    upstream = None if is_detached else _upstream(path)
    ahead, behind = _ahead_behind(path, upstream)
    return WorktreeHygiene(
        path=path,
        branch=branch,
        head_sha=row.get("head_sha", ""),
        is_current=Path(path).resolve() == current_path.resolve(),
        is_detached=is_detached,
        dirty_entries=dirty_entries,
        upstream=upstream,
        ahead=ahead,
        behind=behind,
        issues=_classify_issues(
            branch=branch,
            is_detached=is_detached,
            dirty_entries=dirty_entries,
            upstream=upstream,
            ahead=ahead,
            behind=behind,
        ),
    )


def build_snapshot(
    *,
    include_all_worktrees: bool = False,
    worktree_limit: int = 40,
    filters: list[str] | None = None,
    include_repo_status: bool = False,
) -> HygieneSnapshot:
    """Build a read-only worktree hygiene snapshot."""
    errors: list[str] = []
    filter_values = filters or []
    current_path = Path.cwd().resolve()
    result = _run_command(["git", "worktree", "list", "--porcelain"])
    if result.returncode != 0:
        errors.append("failed to list worktrees")
        parsed: list[dict[str, str]] = []
    else:
        parsed = _parse_worktree_porcelain(result.stdout)

    filtered = [row for row in parsed if _matches_filters(row, filter_values)]
    selected = filtered if include_all_worktrees else filtered[:worktree_limit]
    worktrees = [_build_row(row, current_path) for row in selected]
    current_worktree = next(
        (row.get("path") for row in parsed if Path(row.get("path", "")).resolve() == current_path),
        None,
    )
    issue_counts: dict[str, int] = {}
    for row in worktrees:
        for issue in row.issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1

    return HygieneSnapshot(
        schema=SCHEMA_VERSION,
        current_worktree=current_worktree,
        total_worktrees=len(parsed),
        included_worktrees=len(worktrees),
        worktrees_truncated=len(filtered) > len(selected),
        filters=filter_values,
        issue_counts=issue_counts,
        repo_status=_repo_status() if include_repo_status else None,
        worktrees=worktrees,
        errors=errors,
    )


def format_human(snapshot: HygieneSnapshot) -> str:
    """Format snapshot as human-readable text."""
    lines = [
        f"Worktree Hygiene Snapshot (schema: {snapshot.schema})",
        f"  Current: {snapshot.current_worktree or 'N/A'}",
        f"  Total worktrees: {snapshot.total_worktrees}",
        f"  Included worktrees: {snapshot.included_worktrees}",
        f"  Truncated: {snapshot.worktrees_truncated}",
    ]
    if snapshot.filters:
        lines.append(f"  Filters: {', '.join(snapshot.filters)}")
    if snapshot.issue_counts:
        counts = ", ".join(f"{key}={value}" for key, value in sorted(snapshot.issue_counts.items()))
        lines.append(f"  Issue counts: {counts}")
    if snapshot.repo_status:
        repo = snapshot.repo_status
        lines.append(
            "  Repo status: "
            f"{repo.branch_status}; dirty={repo.dirty_entries}; ahead={repo.ahead}; behind={repo.behind}"
        )
    if snapshot.worktrees:
        lines.append("  Worktrees:")
        for row in snapshot.worktrees:
            issues = f" [{', '.join(row.issues)}]" if row.issues else ""
            branch = row.branch or "detached"
            drift = ""
            if row.ahead is not None or row.behind is not None:
                drift = f" ahead={row.ahead} behind={row.behind}"
            lines.append(f"    - {branch}: {row.path}{drift}{issues}")
    if snapshot.errors:
        lines.append("  Errors:")
        for error in snapshot.errors:
            lines.append(f"    - {error}")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument(
        "--include-all-worktrees",
        action="store_true",
        help="Include all matching worktrees without applying --worktree-limit.",
    )
    parser.add_argument(
        "--worktree-limit", type=int, default=40, help="Maximum worktrees to include."
    )
    parser.add_argument(
        "--filter",
        dest="filters",
        action="append",
        default=[],
        help="Branch or path substring filter. May repeat.",
    )
    parser.add_argument(
        "--repo-status", action="store_true", help="Include current checkout status."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    try:
        snapshot = build_snapshot(
            include_all_worktrees=args.include_all_worktrees,
            worktree_limit=args.worktree_limit,
            filters=args.filters,
            include_repo_status=args.repo_status,
        )
    except Exception as exc:
        print(f"ERROR building worktree hygiene snapshot: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(asdict(snapshot), indent=2, sort_keys=True))
    else:
        print(format_human(snapshot))
    return 0 if not snapshot.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
