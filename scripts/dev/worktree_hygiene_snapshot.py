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
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

SCHEMA_VERSION = "worktree_hygiene_snapshot.v1"
RETIREMENT_SCHEMA_VERSION = "worktree_retirement_plan.v2"
ISSUE_REFERENCE_RE = re.compile(r"(?i)(?:#|issue[-_ /:#]+)(?P<number>[1-9][0-9]*)\b")
PROTECTED_BRANCHES = frozenset({"main", "master"})
PULL_REQUEST_QUERY_LIMIT = 500
ARTIFACT_QUERY_LIMIT = 50
PULL_REQUEST_INVENTORY_TRUNCATED = (
    "pull-request coverage inventory truncated at bounded query limit"
)
TRACKED_DURABLE_PATHS = (
    "docs/context/evidence/**",
    "docs/evidence/**",
    "evidence/**",
    "manifests/**",
    "**/*manifest*",
)
CACHE_ROOTS = frozenset(
    {
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
    }
)
DISPOSABLE_OUTPUT_PREFIXES = (
    "output/coverage/",
    "output/validation/pr_ready/",
    "output/cache/",
    "output/scratch/",
    "output/tmp/",
)
DURABLE_ARTIFACT_TERMS = (
    "artifact",
    "checkpoint",
    "evidence",
    "manifest",
)

RetirementLookup = Callable[["WorktreeHygiene"], "LookupState"]
ArtifactLookup = Callable[["WorktreeHygiene"], list["ArtifactRootInspection"]]

RETIREMENT_PRESERVE = "preserve"
RETIREMENT_REVIEW = "review"
RETIREMENT_REMOVABLE = "removable"
RETIREMENT_PLAN_COMPLETE = "complete"
RETIREMENT_PLAN_INCOMPLETE = "incomplete"
RETIREMENT_PLAN_NEEDS_REVIEW = "needs_review"
DEFAULT_RETIREMENT_WORKTREE_BUDGET = 256
DEFAULT_RETIREMENT_TIME_BUDGET_SECONDS = 60


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


@dataclass(frozen=True, slots=True)
class IgnoredArtifact:
    """A bounded classification of one ignored root in a worktree."""

    path: str
    category: str
    reason: str


@dataclass(frozen=True, slots=True)
class RetirementAssessment:
    """Preservation-aware retirement decision for one worktree."""

    path: str
    branch: str
    head_sha: str
    decision: str  # "preserve", "review", or "removeable"
    coverage: str
    reasons: list[str] = field(default_factory=list)
    issue_numbers: list[int] = field(default_factory=list)
    active_claims: list[int] = field(default_factory=list)
    ignored_artifacts: list[IgnoredArtifact] = field(default_factory=list)
    tracked_durable_paths: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class RetirementEvidence:
    """Evidence inputs used by one retirement assessment."""

    pull_requests: list[dict[str, Any]] = field(default_factory=list)
    pull_request_error: str | None = None
    active_claims: Mapping[int, str] = field(default_factory=dict)
    claims_error: str | None = None
    ignored_artifacts: list[IgnoredArtifact] = field(default_factory=list)
    ignored_artifact_error: str | None = None
    tracked_durable_paths: list[str] = field(default_factory=list)
    tracked_durable_error: str | None = None
    coverage_override: tuple[str, list[str]] | None = None


@dataclass(frozen=True, slots=True)
class RetirementProgress:
    """Machine-readable completion and budget counters for retirement planning."""

    terminal_status: str
    total_worktrees: int
    selected_worktrees: int
    processed_worktrees: int
    unprocessed_worktrees: int
    worktree_budget: int | None
    time_budget_seconds: float | None
    branch_lookup_calls: int
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class RetirementInventory:
    """Bounded local inventory used to build a retirement plan."""

    total_worktrees: int
    current_worktree: str | None
    worktrees_truncated: bool
    rows: list[WorktreeHygiene]
    skipped: list[tuple[WorktreeHygiene, str]]
    errors: list[str]


@dataclass(frozen=True, slots=True)
class RetirementPlan:
    """Read-only preservation-aware worktree retirement plan."""

    schema: str
    total_worktrees: int
    included_worktrees: int
    worktrees_truncated: bool
    current_worktree: str | None
    removeable: list[str]
    preserve: list[str]
    review: list[str]
    worktrees: list[RetirementAssessment]
    errors: list[str]
    progress: RetirementProgress


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
        if not raw_line.strip():
            continue
        status = raw_line[:2]
        path = raw_line[3:] if len(raw_line) > 3 else ""
        if status == "??" and path:
            untracked.append(path)
        elif status == "!!" and path:
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
    if not all_paths:
        if tracked_paths:
            return "tracked_baseline", "clean baseline-tracked files exist under output/"
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


def _unknown_worktree_row(
    row: dict[str, str],
    current_path: Path,
) -> WorktreeHygiene:
    """Represent a row whose local state could not be inspected within the budget."""
    path = row.get("path", "")
    branch = row.get("branch", "")
    return WorktreeHygiene(
        path=path,
        branch=branch,
        head_sha=row.get("head_sha", ""),
        is_current=bool(path) and Path(path).resolve() == current_path.resolve(),
        is_detached=row.get("detached") == "true" or not branch,
        dirty_entries=-1,
        upstream=None,
        ahead=None,
        behind=None,
        issues=["status_unavailable"],
    )


def _build_bounded_worktrees(
    rows: list[dict[str, str]],
    current_path: Path,
    *,
    worktree_budget: int | None,
    deadline: float | None,
) -> tuple[list[WorktreeHygiene], list[tuple[WorktreeHygiene, str]]]:
    """Build worktree rows until a count or wall-clock budget is exhausted."""
    if worktree_budget is not None and worktree_budget < 1:
        raise ValueError("worktree_budget must be at least 1 or None")

    built: list[WorktreeHygiene] = []
    skipped: list[tuple[WorktreeHygiene, str]] = []
    for row in rows:
        if worktree_budget is not None and len(built) >= worktree_budget:
            skipped.append((_unknown_worktree_row(row, current_path), "worktree budget exhausted"))
            continue
        if deadline is not None and time.monotonic() >= deadline:
            skipped.append((_unknown_worktree_row(row, current_path), "time budget exhausted"))
            continue
        built.append(_build_row(row, current_path))
    return built, skipped


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


def _issue_numbers(text: str) -> list[int]:
    """Extract explicit issue references used to associate a worktree claim."""
    return sorted({int(match.group("number")) for match in ISSUE_REFERENCE_RE.finditer(text)})


def _classify_ignored_path(path: str) -> IgnoredArtifact:
    """Classify one ignored path without assuming that unknown output is disposable."""
    normalized = path.strip().rstrip("/")
    lower = normalized.casefold()
    root = lower.split("/", 1)[0]
    if root in CACHE_ROOTS:
        return IgnoredArtifact(normalized, "cache", f"known local cache root: {root}")
    if any(
        lower == prefix.rstrip("/") or lower.startswith(prefix)
        for prefix in DISPOSABLE_OUTPUT_PREFIXES
    ):
        return IgnoredArtifact(
            normalized, "disposable_output", "documented validation or scratch output"
        )
    if any(term in lower.split("/")[-1] for term in DURABLE_ARTIFACT_TERMS):
        return IgnoredArtifact(
            normalized, "durable_required", "ignored artifact name may carry durable evidence"
        )
    if root == "output" or any(term in lower for term in DURABLE_ARTIFACT_TERMS):
        return IgnoredArtifact(
            normalized, "handoff_needed", "ignored output requires explicit human classification"
        )
    return IgnoredArtifact(
        normalized, "handoff_needed", "ignored path is not in the disposable allowlist"
    )


def _ignored_artifacts(path: str) -> tuple[list[IgnoredArtifact], str | None]:
    """Return bounded ignored-root classifications for one worktree."""
    result = _run_command(
        ["git", "status", "--short", "--ignored", "--untracked-files=no"],
        cwd=path,
    )
    if result.returncode != 0:
        return [], f"ignored-artifact status unavailable for {path}: {result.stderr.strip()}"
    artifacts = [
        _classify_ignored_path(line[3:])
        for line in result.stdout.splitlines()
        if line.startswith("!! ") and line[3:].strip()
    ]
    if len(artifacts) > ARTIFACT_QUERY_LIMIT:
        return (
            artifacts[:ARTIFACT_QUERY_LIMIT],
            f"ignored-artifact inventory truncated at {ARTIFACT_QUERY_LIMIT} entries for {path}",
        )
    return artifacts, None


def _tracked_durable_paths(path: str) -> tuple[list[str], str | None]:
    """List changed tracked manifest/evidence paths without scanning committed history."""
    result = _run_command(
        ["git", "diff", "--name-only", "HEAD", "--", *TRACKED_DURABLE_PATHS], cwd=path
    )
    if result.returncode != 0:
        return (
            [],
            f"tracked artifact classification unavailable for {path}: {result.stderr.strip()}",
        )
    paths = sorted({line.strip() for line in result.stdout.splitlines() if line.strip()})
    if len(paths) > ARTIFACT_QUERY_LIMIT:
        return (
            paths[:ARTIFACT_QUERY_LIMIT],
            f"tracked artifact inventory truncated at {ARTIFACT_QUERY_LIMIT} entries for {path}",
        )
    return paths, None


def _load_pull_request_rows(repo_path: Path) -> tuple[list[dict[str, Any]], str | None]:
    """Read bounded all-state PR metadata for branch coverage classification."""
    result = _run_command(
        [
            "gh",
            "pr",
            "list",
            "--state",
            "all",
            "--limit",
            str(PULL_REQUEST_QUERY_LIMIT + 1),
            "--json",
            "number,state,mergedAt,headRefName,headRefOid,title,body",
        ],
        cwd=str(repo_path),
        timeout=60,
    )
    if result.returncode != 0:
        return (
            [],
            f"pull-request coverage unavailable: {result.stderr.strip() or result.stdout.strip()}",
        )
    try:
        payload = json.loads(result.stdout or "null")
    except json.JSONDecodeError as exc:
        return [], f"pull-request coverage returned invalid JSON: {exc}"
    if not isinstance(payload, list):
        return [], "pull-request coverage response is not a list"
    if len(payload) > PULL_REQUEST_QUERY_LIMIT:
        return [], PULL_REQUEST_INVENTORY_TRUNCATED
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            return [], f"pull-request coverage row {index} is not an object"
        if type(row.get("number")) is not int or not isinstance(row.get("state"), str):
            return [], f"pull-request coverage row {index} is malformed"
        if not isinstance(row.get("headRefName"), str):
            return [], f"pull-request coverage row {index} has no head branch"
        if not isinstance(row.get("headRefOid"), str):
            return [], f"pull-request coverage row {index} has no head commit"
        rows.append(row)
    return rows, None


def _load_active_claims(repo_path: Path) -> tuple[dict[int, str], str | None]:
    """Read remote issue-claim refs without changing them."""
    result = _run_command(
        ["git", "ls-remote", "--heads", "origin", "refs/heads/agent-claims/issue-*"],
        cwd=str(repo_path),
        timeout=60,
    )
    if result.returncode != 0:
        return (
            {},
            f"issue-claim state unavailable: {result.stderr.strip() or result.stdout.strip()}",
        )
    claims: dict[int, str] = {}
    pattern = re.compile(r"^refs/heads/agent-claims/issue-([1-9][0-9]*)$")
    for index, line in enumerate(result.stdout.splitlines()):
        parts = line.split()
        if len(parts) != 2:
            return {}, f"issue-claim response row {index} is malformed"
        match = pattern.fullmatch(parts[1])
        if not match or not re.fullmatch(r"[0-9a-fA-F]{7,64}", parts[0]):
            return {}, f"issue-claim response row {index} is malformed"
        claims[int(match.group(1))] = parts[0]
    return claims, None


def _matching_pull_requests(branch: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return PR rows whose head branch exactly names the worktree branch."""
    return [row for row in rows if row.get("headRefName") == branch]


def _query_head_pull_requests(
    repo_path: Path, branch: str
) -> tuple[list[dict[str, Any]], str | None]:
    """Query bounded PR coverage for one branch when the global inventory is truncated."""
    result = _run_command(
        [
            "gh",
            "pr",
            "list",
            "--head",
            branch,
            "--state",
            "all",
            "--limit",
            str(PULL_REQUEST_QUERY_LIMIT + 1),
            "--json",
            "number,state,mergedAt,headRefName,headRefOid,title,body",
        ],
        cwd=str(repo_path),
        timeout=60,
    )
    if result.returncode != 0:
        return (
            [],
            f"pull-request coverage unavailable for branch {branch!r}: "
            f"{result.stderr.strip() or result.stdout.strip()}",
        )
    try:
        payload = json.loads(result.stdout or "null")
    except json.JSONDecodeError as exc:
        return [], f"pull-request coverage for branch {branch!r} returned invalid JSON: {exc}"
    if not isinstance(payload, list):
        return [], f"pull-request coverage for branch {branch!r} is not a list"
    if len(payload) > PULL_REQUEST_QUERY_LIMIT:
        return [], f"pull-request coverage for branch {branch!r} exceeded bounded query limit"
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            return [], f"pull-request coverage for branch {branch!r} row {index} is malformed"
        if type(row.get("number")) is not int or not isinstance(row.get("state"), str):
            return [], f"pull-request coverage for branch {branch!r} row {index} is malformed"
        if row.get("headRefName") != branch or not isinstance(row.get("headRefOid"), str):
            return [], f"pull-request coverage for branch {branch!r} row {index} has invalid head"
        rows.append(row)
    return rows, None


def _pull_request_coverage(
    row: WorktreeHygiene, pull_requests: list[dict[str, Any]]
) -> tuple[str | None, list[str]]:
    """Return exact-head PR coverage, leaving ancestry as a separate fallback."""
    matches = _matching_pull_requests(row.branch, pull_requests)
    exact_matches = [pr for pr in matches if pr.get("headRefOid") == row.head_sha]
    if exact_matches:
        states = {str(pr.get("state") or "").upper() for pr in exact_matches}
        for state, coverage, reasons in (
            ("OPEN", "open_pr", ["open_pr"]),
            ("MERGED", "merged_pr", []),
            ("CLOSED", "closed_pr", ["closed_pr_requires_review"]),
        ):
            if state in states:
                return coverage, reasons
        return "unavailable", ["unknown_pull_request_state"]
    return None, ["branch_head_mismatch"] if matches else []


def _coverage_for_row(
    row: WorktreeHygiene,
    *,
    pull_requests: list[dict[str, Any]],
    pull_request_error: str | None,
) -> tuple[str, list[str]]:
    """Determine whether a branch has authoritative merged/closed coverage."""
    if pull_request_error:
        return "unavailable", ["pull_request_state_unavailable"]
    if not row.branch or not row.head_sha:
        return "unavailable", ["branch_coverage_unavailable"]
    pull_request_coverage, pull_request_reasons = _pull_request_coverage(row, pull_requests)
    if pull_request_coverage:
        return pull_request_coverage, pull_request_reasons

    result = _run_command(
        ["git", "merge-base", "--is-ancestor", row.head_sha, "origin/main"],
        cwd=row.path,
    )
    if result.returncode == 0:
        return "ancestor_of_origin_main", []
    if result.returncode == 1:
        return "unmerged", [*pull_request_reasons, "no_merged_coverage"]
    return "unavailable", ["merge_coverage_unavailable"]


def _base_retirement_reasons(row: WorktreeHygiene) -> tuple[list[str], list[str]]:
    """Return preservation and review reasons derived from local Git state."""
    hard: list[str] = []
    review: list[str] = []
    if row.is_current:
        hard.append("current_worktree")
    if row.branch.casefold() in PROTECTED_BRANCHES:
        hard.append("protected_canonical_branch")
    if row.is_detached:
        hard.append("detached")
    if row.dirty_entries < 0:
        review.append("status_unavailable")
    elif row.dirty_entries > 0:
        hard.append("dirty")
    if row.branch and row.upstream is None:
        hard.append("missing_upstream")
    if row.ahead is None and row.upstream is not None and not row.is_detached:
        review.append("drift_unavailable")
    elif row.ahead is not None and row.ahead > 0:
        hard.append("ahead_commits")
    if row.behind is not None and row.behind > 0:
        review.append("behind_upstream")
    return hard, review


def _artifact_retirement_reasons(evidence: RetirementEvidence) -> tuple[list[str], list[str]]:
    """Return preservation and review reasons derived from artifact evidence."""
    hard: list[str] = []
    review: list[str] = []
    if evidence.ignored_artifact_error:
        review.append("ignored_artifact_classification_unavailable")
    if any(
        item.category in {"durable_required", "handoff_needed"}
        for item in evidence.ignored_artifacts
    ):
        hard.append("ignored_durable_or_unclassified_artifact")
    if evidence.tracked_durable_error:
        review.append("tracked_artifact_classification_unavailable")
    if evidence.tracked_durable_paths:
        hard.append("tracked_durable_evidence_or_manifest")
    return hard, review


def _claim_retirement_reasons(
    issue_numbers: list[int], evidence: RetirementEvidence
) -> tuple[list[str], list[str], list[int]]:
    """Return claim-related reasons and the active issue numbers."""
    active = sorted(set(issue_numbers).intersection(evidence.active_claims))
    if evidence.claims_error:
        return [], ["active_claim_state_unavailable"], active
    if active:
        return ["active_issue_claim"], [], active
    if evidence.active_claims and not issue_numbers:
        return [], ["claim_association_unavailable"], active
    return [], [], active


def _issue_numbers_for_row(row: WorktreeHygiene, pull_requests: list[dict[str, Any]]) -> list[int]:
    """Collect issue references from a branch and its matching PRs."""
    matching = _matching_pull_requests(row.branch, pull_requests)
    numbers = set(_issue_numbers(row.branch))
    numbers.update(
        issue
        for pr in matching
        for issue in _issue_numbers(f"{pr.get('title', '')} {pr.get('body', '')}")
    )
    return sorted(numbers)


def assess_retirement(
    row: WorktreeHygiene,
    evidence: RetirementEvidence | None = None,
) -> RetirementAssessment:
    """Classify a row conservatively as preserve, review, or removeable."""
    evidence = evidence or RetirementEvidence()
    issue_numbers = _issue_numbers_for_row(row, evidence.pull_requests)
    hard_reasons, review_reasons = _base_retirement_reasons(row)
    artifact_hard, artifact_review = _artifact_retirement_reasons(evidence)
    claim_hard, claim_review, active_claim_numbers = _claim_retirement_reasons(
        issue_numbers, evidence
    )
    hard_reasons.extend(artifact_hard)
    hard_reasons.extend(claim_hard)
    review_reasons.extend(artifact_review)
    review_reasons.extend(claim_review)

    coverage, coverage_reasons = evidence.coverage_override or _coverage_for_row(
        row,
        pull_requests=evidence.pull_requests,
        pull_request_error=evidence.pull_request_error,
    )
    review_reasons.extend(coverage_reasons)
    decision = "preserve" if hard_reasons else "review" if review_reasons else "removeable"
    return RetirementAssessment(
        path=row.path,
        branch=row.branch,
        head_sha=row.head_sha,
        decision=decision,
        coverage=coverage,
        reasons=list(dict.fromkeys(hard_reasons + review_reasons)),
        issue_numbers=issue_numbers,
        active_claims=active_claim_numbers,
        ignored_artifacts=evidence.ignored_artifacts,
        tracked_durable_paths=evidence.tracked_durable_paths,
    )


def _unprocessed_retirement_assessment(
    row: WorktreeHygiene,
    reason: str,
) -> RetirementAssessment:
    """Return a review-only assessment for a row skipped by the scan budget."""
    return RetirementAssessment(
        path=row.path,
        branch=row.branch,
        head_sha=row.head_sha,
        decision=RETIREMENT_REVIEW,
        coverage="unavailable",
        reasons=[reason],
        issue_numbers=_issue_numbers(row.branch),
    )


def _retirement_pull_request_context(
    repo_path: Path,
    pull_requests: list[dict[str, Any]] | None,
    pull_request_error: str | None,
) -> tuple[list[dict[str, Any]], str | None, bool, list[str]]:
    """Load global PR context and identify whether per-branch fallback is needed."""
    queried = pull_requests is None and pull_request_error is None
    if queried:
        pull_requests, pull_request_error = _load_pull_request_rows(repo_path)
    use_branch_fallback = queried and pull_request_error == PULL_REQUEST_INVENTORY_TRUNCATED
    errors = [] if use_branch_fallback or not pull_request_error else [pull_request_error]
    return pull_requests or [], pull_request_error, use_branch_fallback, errors


def _pull_requests_for_row(
    repo_path: Path,
    row: WorktreeHygiene,
    *,
    pull_requests: list[dict[str, Any]],
    pull_request_error: str | None,
    use_branch_fallback: bool,
    cache: dict[str, tuple[list[dict[str, Any]], str | None]],
) -> tuple[list[dict[str, Any]], str | None, list[str]]:
    """Return PR evidence for one row, using a cached head query after truncation."""
    if not use_branch_fallback:
        return pull_requests, pull_request_error, []
    if row.branch not in cache:
        cache[row.branch] = (
            _query_head_pull_requests(repo_path, row.branch) if row.branch else ([], None)
        )
    row_pull_requests, row_error = cache[row.branch]
    errors = [f"{row_error} (worktree {row.path})"] if row_error else []
    return row_pull_requests, row_error, errors


def _retirement_evidence_for_row(
    row: WorktreeHygiene,
    *,
    pull_requests: list[dict[str, Any]],
    pull_request_error: str | None,
    active_claims: dict[int, str],
    claims_error: str | None,
) -> tuple[RetirementEvidence, list[str]]:
    """Collect bounded local artifact evidence for one retirement row."""
    ignored, ignored_error = _ignored_artifacts(row.path)
    tracked, tracked_error = _tracked_durable_paths(row.path)
    errors = [error for error in (ignored_error, tracked_error) if error]
    return (
        RetirementEvidence(
            pull_requests=pull_requests,
            pull_request_error=pull_request_error,
            active_claims=active_claims,
            claims_error=claims_error,
            ignored_artifacts=ignored,
            ignored_artifact_error=ignored_error,
            tracked_durable_paths=tracked,
            tracked_durable_error=tracked_error,
        ),
        errors,
    )


def _prepare_retirement_inventory(
    *,
    include_all_worktrees: bool,
    worktree_limit: int,
    filters: list[str],
    snapshot: HygieneSnapshot | None,
    worktree_budget: int | None,
    deadline: float | None,
) -> RetirementInventory:
    """Build the bounded local portion of a retirement plan."""
    if snapshot is not None:
        rows = list(snapshot.worktrees)
        skipped: list[tuple[WorktreeHygiene, str]] = []
        if worktree_budget is not None and len(rows) > worktree_budget:
            skipped = [(row, "worktree budget exhausted") for row in rows[worktree_budget:]]
            rows = rows[:worktree_budget]
        if deadline is not None and time.monotonic() >= deadline:
            skipped.extend((row, "time budget exhausted") for row in rows)
            rows = []
        return RetirementInventory(
            total_worktrees=snapshot.total_worktrees,
            current_worktree=snapshot.current_worktree,
            worktrees_truncated=snapshot.worktrees_truncated,
            rows=rows,
            skipped=skipped,
            errors=list(snapshot.errors),
        )

    current_path = Path.cwd().resolve()
    result = _run_command(["git", "worktree", "list", "--porcelain"])
    if result.returncode != 0:
        return RetirementInventory(
            total_worktrees=0,
            current_worktree=None,
            worktrees_truncated=False,
            rows=[],
            skipped=[],
            errors=["failed to list worktrees"],
        )

    parsed = _parse_worktree_porcelain(result.stdout)
    filtered = [row for row in parsed if _matches_filters(row, filters)]
    selected = filtered if include_all_worktrees else filtered[:worktree_limit]
    rows, skipped = _build_bounded_worktrees(
        selected,
        current_path,
        worktree_budget=worktree_budget,
        deadline=deadline,
    )
    current_worktree = next(
        (
            row["path"]
            for row in parsed
            if row.get("path") and Path(row["path"]).resolve() == current_path
        ),
        None,
    )
    return RetirementInventory(
        total_worktrees=len(parsed),
        current_worktree=current_worktree,
        worktrees_truncated=len(selected) < len(filtered),
        rows=rows,
        skipped=skipped,
        errors=[],
    )


def _classify_retirement_rows(
    *,
    rows: list[WorktreeHygiene],
    skipped: list[tuple[WorktreeHygiene, str]],
    repo_path: Path,
    pull_requests: list[dict[str, Any]] | None,
    pull_request_error: str | None,
    active_claims: dict[int, str] | None,
    claims_error: str | None,
    deadline: float | None,
) -> tuple[list[RetirementAssessment], list[tuple[WorktreeHygiene, str]], list[str], int]:
    """Classify rows while preserving any that exceed the remaining time budget."""
    errors: list[str] = []
    if not rows:
        return (
            [_unprocessed_retirement_assessment(row, reason) for row, reason in skipped],
            skipped,
            [reason for _, reason in skipped],
            0,
        )
    (
        pull_requests,
        pull_request_error,
        use_branch_fallback,
        pull_request_errors,
    ) = _retirement_pull_request_context(repo_path, pull_requests, pull_request_error)
    if active_claims is None and claims_error is None:
        active_claims, claims_error = _load_active_claims(repo_path)
    errors.extend(pull_request_errors)
    if claims_error:
        errors.append(claims_error)

    cache: dict[str, tuple[list[dict[str, Any]], str | None]] = {}
    assessments_by_path: dict[str, RetirementAssessment] = {}
    assessed_rows: list[WorktreeHygiene] = []
    branch_lookup_calls = 0
    for index, row in enumerate(rows):
        if deadline is not None and time.monotonic() >= deadline:
            skipped.extend((pending, "time budget exhausted") for pending in rows[index:])
            break
        had_cached_branch = row.branch in cache
        row_pull_requests, row_pull_request_error, row_pr_errors = _pull_requests_for_row(
            repo_path,
            row,
            pull_requests=pull_requests,
            pull_request_error=pull_request_error,
            use_branch_fallback=use_branch_fallback,
            cache=cache,
        )
        if use_branch_fallback and row.branch and not had_cached_branch:
            branch_lookup_calls += 1
        evidence, evidence_errors = _retirement_evidence_for_row(
            row,
            pull_requests=row_pull_requests,
            pull_request_error=row_pull_request_error,
            active_claims=active_claims or {},
            claims_error=claims_error,
        )
        errors.extend(row_pr_errors)
        errors.extend(evidence_errors)
        assessments_by_path[row.path] = assess_retirement(row, evidence)
        assessed_rows.append(row)

    for row, reason in skipped:
        assessments_by_path[row.path] = _unprocessed_retirement_assessment(row, reason)
        errors.append(reason)
    ordered_rows = [*assessed_rows, *(row for row, _ in skipped)]
    return (
        [assessments_by_path[row.path] for row in ordered_rows],
        skipped,
        errors,
        branch_lookup_calls,
    )


def build_retirement_plan(  # noqa: PLR0913 - preserves the public injected-state contract
    *,
    include_all_worktrees: bool = False,
    worktree_limit: int = 40,
    worktree_budget: int | None = DEFAULT_RETIREMENT_WORKTREE_BUDGET,
    time_budget_seconds: float | None = DEFAULT_RETIREMENT_TIME_BUDGET_SECONDS,
    filters: list[str] | None = None,
    snapshot: HygieneSnapshot | None = None,
    pull_requests: list[dict[str, Any]] | None = None,
    pull_request_error: str | None = None,
    active_claims: dict[int, str] | None = None,
    claims_error: str | None = None,
) -> RetirementPlan:
    """Build a read-only preservation-aware retirement projection.

    The retirement path owns its inventory construction so ``--include-all-worktrees`` cannot
    spend an unbounded amount of time building an ordinary snapshot before the retirement budget
    is applied. Rows that do not fit the budget are retained as review-only assessments.
    """
    if worktree_budget is not None and worktree_budget < 1:
        raise ValueError("worktree_budget must be at least 1 or None")
    if time_budget_seconds is not None and time_budget_seconds < 0:
        raise ValueError("time_budget_seconds must be non-negative or None")

    started = time.monotonic()
    deadline = started + time_budget_seconds if time_budget_seconds is not None else None
    inventory = _prepare_retirement_inventory(
        include_all_worktrees=include_all_worktrees,
        worktree_limit=worktree_limit,
        filters=filters or [],
        snapshot=snapshot,
        worktree_budget=worktree_budget,
        deadline=deadline,
    )
    rows = inventory.rows
    skipped = list(inventory.skipped)
    if deadline is not None and time.monotonic() >= deadline:
        skipped.extend((row, "time budget exhausted") for row in rows)
        rows = []

    assessments, skipped, classification_errors, branch_lookup_calls = _classify_retirement_rows(
        rows=rows,
        skipped=skipped,
        repo_path=Path.cwd().resolve(),
        pull_requests=pull_requests,
        pull_request_error=pull_request_error,
        active_claims=active_claims,
        claims_error=claims_error,
        deadline=deadline,
    )
    errors = list(inventory.errors) + classification_errors
    if inventory.worktrees_truncated:
        errors.append(
            "worktree inventory truncated; use --include-all-worktrees for complete planning"
        )
    if skipped:
        errors.append("retirement scan incomplete: unprocessed worktrees are review-only")
    selected_count = len(assessments)
    terminal_status = (
        RETIREMENT_PLAN_INCOMPLETE
        if skipped
        else RETIREMENT_PLAN_NEEDS_REVIEW
        if errors
        else RETIREMENT_PLAN_COMPLETE
    )
    elapsed = round(time.monotonic() - started, 3)
    return RetirementPlan(
        schema=RETIREMENT_SCHEMA_VERSION,
        total_worktrees=inventory.total_worktrees,
        included_worktrees=selected_count,
        worktrees_truncated=inventory.worktrees_truncated,
        current_worktree=inventory.current_worktree,
        removeable=[row.path for row in assessments if row.decision == "removeable"],
        preserve=[row.path for row in assessments if row.decision == "preserve"],
        review=[row.path for row in assessments if row.decision == "review"],
        worktrees=list(assessments),
        errors=list(dict.fromkeys(errors)),
        progress=RetirementProgress(
            terminal_status=terminal_status,
            total_worktrees=inventory.total_worktrees,
            selected_worktrees=selected_count,
            processed_worktrees=selected_count - len(skipped),
            unprocessed_worktrees=len(skipped),
            worktree_budget=worktree_budget,
            time_budget_seconds=time_budget_seconds,
            branch_lookup_calls=branch_lookup_calls,
            elapsed_seconds=elapsed,
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


def format_retirement_plan(plan: RetirementPlan) -> str:
    """Format a preservation-aware retirement plan without offering deletion."""
    lines = [
        f"Worktree Retirement Plan (schema: {plan.schema})",
        f"  Total worktrees: {plan.total_worktrees}",
        f"  Included worktrees: {plan.included_worktrees}",
        f"  Truncated: {plan.worktrees_truncated}",
        f"  Terminal status: {plan.progress.terminal_status}",
        f"  Removeable: {len(plan.removeable)}",
        f"  Preserve: {len(plan.preserve)}",
        f"  Review: {len(plan.review)}",
        "  Progress: "
        f"selected={plan.progress.selected_worktrees} "
        f"processed={plan.progress.processed_worktrees} "
        f"unprocessed={plan.progress.unprocessed_worktrees} "
        f"worktree_budget={plan.progress.worktree_budget} "
        f"time_budget_s={plan.progress.time_budget_seconds} "
        f"branch_lookup_calls={plan.progress.branch_lookup_calls} "
        f"elapsed_s={plan.progress.elapsed_seconds}",
    ]
    for row in plan.worktrees:
        lines.append(
            f"  - [{row.decision.upper()}] {row.branch or 'detached'}: {row.path}"
            f" coverage={row.coverage}"
        )
        if row.reasons:
            lines.append(f"    reasons: {', '.join(row.reasons)}")
        if row.ignored_artifacts:
            categories = ", ".join(f"{item.path}={item.category}" for item in row.ignored_artifacts)
            lines.append(f"    ignored: {categories}")
        if row.tracked_durable_paths:
            lines.append(f"    tracked durable paths: {', '.join(row.tracked_durable_paths)}")
    if plan.errors:
        lines.append("  Errors:")
        lines.extend(f"    - {error}" for error in plan.errors)
    lines.append("  No worktrees were removed; this command is read-only.")
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
            "Include a read-only preservation-aware retirement plan/projection. "
            "This never deletes worktrees."
        ),
    )
    parser.add_argument(
        "--worktree-budget",
        type=int,
        default=DEFAULT_RETIREMENT_WORKTREE_BUDGET,
        help=(
            "Maximum number of worktrees to fully inspect in a retirement plan. "
            "Rows beyond this budget are review-only."
        ),
    )
    parser.add_argument(
        "--time-budget-seconds",
        type=float,
        default=DEFAULT_RETIREMENT_TIME_BUDGET_SECONDS,
        help=(
            "Maximum wall-clock seconds for retirement inventory and classification. "
            "Rows beyond this budget are review-only."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    try:
        if args.retirement_plan:
            plan = build_retirement_plan(
                include_all_worktrees=args.include_all_worktrees,
                worktree_limit=args.worktree_limit,
                worktree_budget=args.worktree_budget,
                time_budget_seconds=args.time_budget_seconds,
                filters=args.filters,
            )
            if args.json:
                print(json.dumps(asdict(plan), indent=2, sort_keys=True))
            else:
                print(format_retirement_plan(plan))
            return 0 if not plan.errors else 1
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
