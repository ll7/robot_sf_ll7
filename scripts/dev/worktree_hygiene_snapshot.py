#!/usr/bin/env python3
"""Emit a compact worktree hygiene snapshot.

This helper is read-only. It summarizes branch drift, dirty worktrees, detached
heads, and missing upstreams without printing full `git worktree` output. Use it
before remote maintenance, stale-worktree cleanup planning, or broad PR loops.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCHEMA_VERSION = "worktree_hygiene_snapshot.v1"

RetirementLookup = Callable[["WorktreeHygiene"], "LookupState"]
ArtifactLookup = Callable[["WorktreeHygiene"], list["ArtifactRootInspection"]]

RETIREMENT_PRESERVE = "preserve"
RETIREMENT_REVIEW = "review"
RETIREMENT_REMOVABLE = "removable"


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
    retirement: RetirementProjection | None = None


@dataclass(frozen=True, slots=True)
class ArtifactRootInspection:
    """Read-only classification of a generated or ignored root."""

    root: str
    classification: str
    status: str
    ignored_entries: int = 0
    untracked_entries: int = 0
    tracked_entries: int = 0
    sample_paths: list[str] = field(default_factory=list)
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class LookupState:
    """Injectable external or local state used by retirement projection."""

    status: str
    refs: list[str] = field(default_factory=list)
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class RetirementProjection:
    """Read-only worktree retirement recommendation."""

    action: str
    reasons: list[str]
    claim_state: LookupState
    merge_state: LookupState
    artifact_roots: list[ArtifactRootInspection]


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

        new_row = _parse_worktree_line(line, current)
        if new_row is not current:
            if current:
                worktrees.append(current)
            current = new_row

    if current:
        worktrees.append(current)

    return worktrees


def _parse_worktree_line(line: str, current: dict[str, str]) -> dict[str, str]:
    """Apply one porcelain line to the current worktree row."""
    parts = line.split(" ", 1)
    if len(parts) != 2:
        if line == "detached":
            current["detached"] = "true"
        return current

    key, value = parts
    if key == "worktree":
        return {"path": value}
    if key == "HEAD":
        current["head_sha"] = value
    elif key == "branch":
        current["branch"] = value.removeprefix("refs/heads/")
    return current


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


def _sample_paths(paths: list[str], *, limit: int = 8) -> list[str]:
    """Return a compact deterministic path sample."""
    return sorted(set(paths))[:limit]


def _status_paths(stdout: str) -> tuple[list[str], list[str], list[str]]:
    """Split short-status output into tracked-dirty, untracked, and ignored paths."""
    tracked_dirty: list[str] = []
    untracked: list[str] = []
    ignored: list[str] = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        path = line[3:] if len(line) > 3 else ""
        if line.startswith("?? ") and path:
            untracked.append(path)
        elif line.startswith("!! ") and path:
            ignored.append(path)
        elif path:
            tracked_dirty.append(path)
    return tracked_dirty, untracked, ignored


def _classify_output_root(
    *,
    tracked_paths: list[str],
    baseline_tracked_paths: list[str],
    tracked_dirty_paths: list[str],
    untracked_paths: list[str],
    ignored_paths: list[str],
) -> tuple[str, str]:
    """Classify output-like roots without deleting or moving content."""
    all_paths = untracked_paths + ignored_paths
    if tracked_dirty_paths:
        return "tracked_evidence", "tracked output/ files have local modifications"
    if set(tracked_paths) - set(baseline_tracked_paths):
        return "tracked_evidence", "branch-local tracked files exist under output/"
    if tracked_paths:
        return "tracked_baseline", "clean baseline-tracked files exist under output/"
    if not all_paths:
        return "none", "output/ is absent or empty"

    durable_markers = (
        "manifest",
        "sha256sums",
        "checksum",
        "evidence",
        "dossier",
        "registry",
        "report",
        "summary",
    )
    durable_suffixes = (".jsonl", ".parquet", ".csv", ".mp4", ".webm", ".png", ".pdf")
    durable_dirs = (
        "output/benchmarks/",
        "output/recordings/",
        "output/research_reports/",
        "output/run-tracker/",
        "output/figures/",
    )
    cache_dirs = ("output/model_cache/", "output/wandb/", "output/tmp/")
    disposable_dirs = ("output/coverage/", "output/validation/pr_ready/")

    lowered = [path.lower() for path in all_paths]
    if any(path.startswith(durable_dirs) for path in lowered) or any(
        marker in path or path.endswith(durable_suffixes)
        for path in lowered
        for marker in durable_markers
    ):
        return "durable_required", "output/ contains evidence-like paths"
    if all(path.startswith(cache_dirs) for path in lowered):
        return "ignored_cache", "output/ contains only recognized cache paths"
    if all(path.startswith(disposable_dirs) for path in lowered):
        return "disposable_output", "output/ contains only recognized validation leftovers"
    return "handoff_needed", "output/ contents need owner classification"


def _inspect_output_root(row: WorktreeHygiene) -> list[ArtifactRootInspection]:
    """Inspect ignored/untracked output content without deleting or moving it."""
    path = row.path
    if not path:
        return [
            ArtifactRootInspection(
                root="output",
                classification="unavailable",
                status="unavailable",
                reason="worktree path unavailable",
            )
        ]

    status = _run_command(
        ["git", "status", "--ignored", "--short", "-uall", "--", "output"],
        cwd=path,
    )
    if status.returncode != 0:
        return [
            ArtifactRootInspection(
                root="output",
                classification="unavailable",
                status="unavailable",
                reason="git status for output/ failed",
            )
        ]

    tracked = _run_command(["git", "ls-files", "--", "output"], cwd=path)
    if tracked.returncode != 0:
        return [
            ArtifactRootInspection(
                root="output",
                classification="unavailable",
                status="unavailable",
                reason="git ls-files for output/ failed",
            )
        ]

    tracked_dirty_paths, untracked_paths, ignored_paths = _status_paths(status.stdout)
    tracked_paths = [line for line in tracked.stdout.splitlines() if line.strip()]
    baseline_tracked_paths: list[str] = []
    if tracked_paths:
        baseline = _run_command(
            ["git", "ls-tree", "-r", "--name-only", "origin/main", "--", "output"],
            cwd=path,
        )
        if baseline.returncode != 0:
            return [
                ArtifactRootInspection(
                    root="output",
                    classification="unavailable",
                    status="unavailable",
                    tracked_entries=len(tracked_paths),
                    sample_paths=_sample_paths(tracked_paths),
                    reason="git ls-tree for baseline output/ failed",
                )
            ]
        baseline_tracked_paths = [line for line in baseline.stdout.splitlines() if line.strip()]
    classification, reason = _classify_output_root(
        tracked_paths=tracked_paths,
        baseline_tracked_paths=baseline_tracked_paths,
        tracked_dirty_paths=tracked_dirty_paths,
        untracked_paths=untracked_paths,
        ignored_paths=ignored_paths,
    )
    return [
        ArtifactRootInspection(
            root="output",
            classification=classification,
            status="ok",
            ignored_entries=len(ignored_paths),
            untracked_entries=len(untracked_paths),
            tracked_entries=len(tracked_paths),
            sample_paths=_sample_paths(tracked_paths + untracked_paths + ignored_paths),
            reason=reason,
        )
    ]


_ISSUE_REF_RE = re.compile(r"(?:^|[-_/])(?:issue|gh)-(?P<number>\d+)(?:$|[-_/])")


def _default_claim_state(row: WorktreeHygiene) -> LookupState:
    """Conservative local-only issue-claim state.

    Rows with issue-like branch/path names require review because a live claim check is outside this
    read-only local helper. Rows without an issue-like signal are treated as having no local claim.
    """
    haystack = " ".join((row.branch, row.path)).lower()
    match = _ISSUE_REF_RE.search(haystack)
    if match:
        return LookupState(
            status="unavailable",
            refs=[f"issue-{match.group('number')}"],
            reason="issue-like claim reference requires live or injected claim review",
        )
    return LookupState(status="inactive", reason="no issue-like local claim reference")


def _default_merge_state(row: WorktreeHygiene) -> LookupState:
    """Return local merged-to-origin/main state without network access."""
    if not row.path:
        return LookupState(status="unavailable", reason="worktree path unavailable")
    result = _run_command(
        ["git", "merge-base", "--is-ancestor", "HEAD", "origin/main"], cwd=row.path
    )
    if result.returncode == 0:
        return LookupState(
            status="merged", refs=["origin/main"], reason="HEAD is ancestor of origin/main"
        )
    if result.returncode == 1:
        return LookupState(
            status="unmerged", refs=["origin/main"], reason="HEAD is not in origin/main"
        )
    return LookupState(status="unavailable", refs=["origin/main"], reason="merge-base check failed")


def _local_retirement_risks(row: WorktreeHygiene) -> tuple[bool, bool, list[str]]:
    """Return local preserve/review signals from Git status and branch shape."""
    preserve = False
    review = False
    reasons: list[str] = []

    if row.dirty_entries < 0:
        review = True
        reasons.append("status unavailable")
    elif row.dirty_entries > 0:
        preserve = True
        reasons.append("tracked or untracked status is dirty")

    if row.ahead is None:
        review = True
        reasons.append("ahead state unavailable")
    elif row.ahead > 0:
        preserve = True
        reasons.append("worktree has commits ahead of upstream")

    if row.is_detached:
        review = True
        reasons.append("detached HEAD needs human review")
    if row.branch and not row.upstream:
        review = True
        reasons.append("upstream is missing")
    return preserve, review, reasons


def _state_retirement_risks(
    *,
    claim_state: LookupState,
    merge_state: LookupState,
) -> tuple[bool, bool, list[str]]:
    """Return preserve/review signals from injectable claim and merge state."""
    preserve = False
    review = False
    reasons: list[str] = []

    if claim_state.status == "active":
        preserve = True
        reasons.append("active issue claim")
    elif claim_state.status != "inactive":
        review = True
        reasons.append(f"claim state {claim_state.status}")

    if merge_state.status == "unmerged":
        preserve = True
        reasons.append("HEAD is not confirmed merged")
    elif merge_state.status != "merged":
        review = True
        reasons.append(f"merge state {merge_state.status}")
    return preserve, review, reasons


def _artifact_retirement_risks(
    artifacts: list[ArtifactRootInspection],
) -> tuple[bool, bool, list[str]]:
    """Return preserve/review signals from artifact root inspection."""
    preserve = False
    review = False
    reasons: list[str] = []

    for artifact in artifacts:
        if artifact.status != "ok" or artifact.classification == "unavailable":
            review = True
            reasons.append(f"{artifact.root} artifact state unavailable")
        elif artifact.classification in {"durable_required", "tracked_evidence", "handoff_needed"}:
            preserve = True
            reasons.append(f"{artifact.root} has {artifact.classification}")
    return preserve, review, reasons


def _build_retirement_projection(
    row: WorktreeHygiene,
    *,
    claim_lookup: RetirementLookup,
    merge_lookup: RetirementLookup,
    artifact_lookup: ArtifactLookup,
) -> RetirementProjection:
    """Classify a row as preserve, review, or removable only when safe."""
    claim_state = claim_lookup(row)
    merge_state = merge_lookup(row)
    artifact_roots = artifact_lookup(row)
    risk_groups = [
        _local_retirement_risks(row),
        _state_retirement_risks(claim_state=claim_state, merge_state=merge_state),
        _artifact_retirement_risks(artifact_roots),
    ]
    preserve = any(group[0] for group in risk_groups)
    review = any(group[1] for group in risk_groups)
    reasons = [reason for group in risk_groups for reason in group[2]]

    if preserve:
        action = RETIREMENT_PRESERVE
    elif review:
        action = RETIREMENT_REVIEW
    else:
        action = RETIREMENT_REMOVABLE
        reasons.append("clean, merged, unclaimed, and no durable artifact signal")

    return RetirementProjection(
        action=action,
        reasons=_sample_paths(reasons, limit=16),
        claim_state=claim_state,
        merge_state=merge_state,
        artifact_roots=artifact_roots,
    )


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
    include_retirement_plan: bool = False,
    claim_lookup: RetirementLookup | None = None,
    merge_lookup: RetirementLookup | None = None,
    artifact_lookup: ArtifactLookup | None = None,
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
    if include_retirement_plan:
        resolved_claim_lookup = claim_lookup or _default_claim_state
        resolved_merge_lookup = merge_lookup or _default_merge_state
        resolved_artifact_lookup = artifact_lookup or _inspect_output_root
        worktrees = [
            WorktreeHygiene(
                path=row.path,
                branch=row.branch,
                head_sha=row.head_sha,
                is_current=row.is_current,
                is_detached=row.is_detached,
                dirty_entries=row.dirty_entries,
                upstream=row.upstream,
                ahead=row.ahead,
                behind=row.behind,
                issues=row.issues,
                retirement=_build_retirement_projection(
                    row,
                    claim_lookup=resolved_claim_lookup,
                    merge_lookup=resolved_merge_lookup,
                    artifact_lookup=resolved_artifact_lookup,
                ),
            )
            for row in worktrees
        ]
    current_worktree = next(
        (
            row["path"]
            for row in parsed
            if row.get("path") and Path(row["path"]).resolve() == current_path
        ),
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


def _format_retirement(row: WorktreeHygiene) -> list[str]:
    """Format row-level retirement details."""
    if not row.retirement:
        return []
    lines: list[str] = []
    reasons = "; ".join(row.retirement.reasons)
    lines.append(f"      retirement: {row.retirement.action} ({reasons})")
    for artifact in row.retirement.artifact_roots:
        samples = f"; samples={', '.join(artifact.sample_paths)}" if artifact.sample_paths else ""
        lines.append(
            "      artifact: "
            f"{artifact.root}={artifact.classification} "
            f"tracked={artifact.tracked_entries} "
            f"untracked={artifact.untracked_entries} "
            f"ignored={artifact.ignored_entries}{samples}"
        )
    return lines


def _format_worktree_row(row: WorktreeHygiene) -> list[str]:
    """Format one worktree row."""
    issues = f" [{', '.join(row.issues)}]" if row.issues else ""
    branch = row.branch or "detached"
    drift = ""
    if row.ahead is not None or row.behind is not None:
        drift = f" ahead={row.ahead} behind={row.behind}"
    return [f"    - {branch}: {row.path}{drift}{issues}", *_format_retirement(row)]


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
            lines.extend(_format_worktree_row(row))
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
    parser.add_argument(
        "--retirement-plan",
        action="store_true",
        help=(
            "Include a read-only preservation-aware retirement projection. "
            "This never deletes worktrees."
        ),
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
            include_retirement_plan=args.retirement_plan,
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
