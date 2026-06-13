#!/usr/bin/env python3
"""Dry-run stale-worktree reaper that classifies cleanup candidates.

Emits a preservation-aware deletion plan without removing anything by default.
Use --apply to actually remove safe candidates; risky candidates are refused
unless explicit safeguards are satisfied.

Default behavior is dry-run only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCHEMA_VERSION = "stale_worktree_reaper.v1"


@dataclass(frozen=True, slots=True)
class WorktreeCandidate:
    """A single worktree cleanup candidate."""

    path: str
    branch: str
    head_sha: str
    is_current: bool
    classification: str  # "current", "clean_stale", "risky"
    risk_flags: list[str] = field(default_factory=list)
    preservation_required: str = ""


@dataclass(frozen=True, slots=True)
class ReaperPlan:
    """Deletion plan output."""

    schema: str
    mode: str  # "dry_run" or "apply"
    total_worktrees: int
    current_worktree: str | None
    candidates: list[WorktreeCandidate]
    deletable: list[str]
    refused: list[str]
    errors: list[str]
    audit_log: list[str]


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
            current["head_sha"] = value
        elif key == "branch":
            current["branch"] = value.removeprefix("refs/heads/")
        elif key in {"bare", "detached"}:
            current[key] = "true"
    if current:
        worktrees.append(current)
    return worktrees


def _is_worktree_dirty(path: str) -> bool:
    """Check if a worktree has uncommitted changes."""
    result = _run_command(["git", "status", "--porcelain"], cwd=path)
    return result.returncode == 0 and bool(result.stdout.strip())


def _has_unpushed_commits(path: str, branch: str) -> bool:
    """Check if a worktree has commits not pushed to its upstream branch."""
    if not branch:
        return True
    upstream = _run_command(["git", "rev-parse", "--abbrev-ref", "@{upstream}"], cwd=path)
    if upstream.returncode != 0 or not upstream.stdout.strip():
        return True
    result = _run_command(
        ["git", "log", f"{upstream.stdout.strip()}..HEAD", "--oneline"],
        cwd=path,
    )
    if result.returncode != 0:
        return True
    return bool(result.stdout.strip())


def _has_open_pr(branch: str) -> bool:
    """Check if the branch has an open PR on GitHub."""
    if not branch:
        return False
    result = _run_command(
        ["gh", "pr", "list", "--head", branch, "--state", "open", "--json", "number"],
    )
    if result.returncode != 0:
        return False
    try:
        data = json.loads(result.stdout)
        return isinstance(data, list) and len(data) > 0
    except (json.JSONDecodeError, TypeError):
        return False


def _has_ignored_output(path: str) -> bool:
    """Check if a worktree has ignored files (e.g. output/ contents)."""
    result = _run_command(
        ["git", "status", "--ignored", "--short", "-uall"],
        cwd=path,
    )
    if result.returncode != 0:
        return False
    for line in result.stdout.splitlines():
        if line.startswith("!! "):
            return True
    return False


def classify_worktree(
    *,
    path: str,
    branch: str,
    head_sha: str,
    current_path: str,
    skip_pr_check: bool = False,
) -> WorktreeCandidate:
    """Classify a single worktree as current, clean_stale, or risky."""
    is_current = Path(path).resolve() == Path(current_path).resolve()
    if is_current:
        return WorktreeCandidate(
            path=path,
            branch=branch,
            head_sha=head_sha,
            is_current=True,
            classification="current",
            preservation_required="current worktree",
        )

    risk_flags: list[str] = []

    if _is_worktree_dirty(path):
        risk_flags.append("dirty")

    if _has_unpushed_commits(path, branch):
        risk_flags.append("unpushed_commits")

    if skip_pr_check and branch:
        risk_flags.append("pr_check_skipped")
    elif _has_open_pr(branch):
        risk_flags.append("open_pr")

    if _has_ignored_output(path):
        risk_flags.append("ignored_output")

    if risk_flags:
        return WorktreeCandidate(
            path=path,
            branch=branch,
            head_sha=head_sha,
            is_current=False,
            classification="risky",
            risk_flags=risk_flags,
            preservation_required=f"risky: {', '.join(risk_flags)}",
        )

    return WorktreeCandidate(
        path=path,
        branch=branch,
        head_sha=head_sha,
        is_current=False,
        classification="clean_stale",
    )


def build_plan(
    *,
    current_path: str | None = None,
    skip_pr_check: bool = False,
    limit: int = 0,
) -> ReaperPlan:
    """Build a deletion plan from current repository worktrees."""
    errors: list[str] = []
    if current_path is None:
        current_path = str(Path.cwd().resolve())

    result = _run_command(["git", "worktree", "list", "--porcelain"])
    if result.returncode != 0:
        errors.append("failed to list worktrees")
        return ReaperPlan(
            schema=SCHEMA_VERSION,
            mode="dry_run",
            total_worktrees=0,
            current_worktree=None,
            candidates=[],
            deletable=[],
            refused=[],
            errors=errors,
            audit_log=["failed to list worktrees"],
        )

    parsed = _parse_worktree_porcelain(result.stdout)
    current_worktree = None
    candidates: list[WorktreeCandidate] = []
    audit_log: list[str] = []

    worktrees_to_check = parsed[:limit] if limit > 0 else parsed
    for wt in worktrees_to_check:
        wt_path = wt.get("path", "")
        branch = wt.get("branch", "")
        head_sha = wt.get("head_sha", "")

        candidate = classify_worktree(
            path=wt_path,
            branch=branch,
            head_sha=head_sha,
            current_path=current_path,
            skip_pr_check=skip_pr_check,
        )
        if candidate.is_current:
            current_worktree = wt_path
        candidates.append(candidate)
        audit_log.append(
            f"classified {wt_path} as {candidate.classification}"
            + (f" ({', '.join(candidate.risk_flags)})" if candidate.risk_flags else "")
        )

    deletable = [c.path for c in candidates if c.classification == "clean_stale"]
    refused = [c.path for c in candidates if c.classification == "risky"]

    return ReaperPlan(
        schema=SCHEMA_VERSION,
        mode="dry_run",
        total_worktrees=len(parsed),
        current_worktree=current_worktree,
        candidates=candidates,
        deletable=deletable,
        refused=refused,
        errors=errors,
        audit_log=audit_log,
    )


def apply_deletions(plan: ReaperPlan, *, force: bool = False) -> ReaperPlan:
    """Apply safe deletions from the plan. Refuses risky candidates."""
    errors = list(plan.errors)

    if plan.errors:
        return plan

    deletable = list(plan.deletable)
    refused = list(plan.refused)
    audit_log = list(plan.audit_log)

    for candidate in plan.candidates:
        if candidate.classification != "clean_stale":
            if candidate.classification == "risky":
                audit_log.append(f"refused risky candidate {candidate.path}")
            continue
        if candidate.is_current:
            refused.append(candidate.path)
            audit_log.append(f"refused current worktree {candidate.path}")
            continue

        result = _run_command(["git", "worktree", "remove", candidate.path])
        if result.returncode != 0:
            errors.append(f"failed to remove {candidate.path}: {result.stderr.strip()}")
            refused.append(candidate.path)
            audit_log.append(f"failed to remove {candidate.path}")
        else:
            deletable = [d for d in deletable if d != candidate.path]
            audit_log.append(f"removed {candidate.path}")

    return ReaperPlan(
        schema=plan.schema,
        mode="apply",
        total_worktrees=plan.total_worktrees,
        current_worktree=plan.current_worktree,
        candidates=plan.candidates,
        deletable=deletable,
        refused=refused,
        errors=errors,
        audit_log=audit_log,
    )


def _append_candidate_lines(lines: list[str], candidates: list[WorktreeCandidate]) -> None:
    """Append human-readable candidate rows."""
    for c in candidates:
        if c.classification == "current":
            tag = "[CURRENT]"
        elif c.classification == "clean_stale":
            tag = "[DELETABLE]"
        else:
            tag = f"[RISKY: {', '.join(c.risk_flags)}]"
        lines.append(f"  {tag} {c.branch or 'detached'}: {c.path}")


def _append_path_section(lines: list[str], title: str, paths: list[str]) -> None:
    """Append a titled path list when non-empty."""
    if paths:
        lines.append("")
        lines.append(f"  {title} ({len(paths)}):")
        for p in paths:
            lines.append(f"    - {p}")


def _append_text_section(lines: list[str], title: str, values: list[str]) -> None:
    """Append a titled text list when non-empty."""
    if values:
        lines.append("")
        lines.append(f"  {title}:")
        for value in values:
            lines.append(f"    - {value}")


def format_human(plan: ReaperPlan) -> str:
    """Format plan as human-readable text."""
    lines = [
        f"Stale Worktree Reaper (schema: {plan.schema})",
        f"  Mode: {plan.mode}",
        f"  Total worktrees: {plan.total_worktrees}",
        f"  Current: {plan.current_worktree or 'N/A'}",
        "",
    ]

    _append_candidate_lines(lines, plan.candidates)
    _append_path_section(lines, "Deletable", plan.deletable)
    _append_path_section(lines, "Refused", plan.refused)
    _append_text_section(lines, "Errors", plan.errors)
    _append_text_section(lines, "Audit log", plan.audit_log)

    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually remove safe worktrees (risky candidates still refused)",
    )
    parser.add_argument(
        "--skip-pr-check",
        action="store_true",
        help="Skip GitHub PR lookup (faster, offline-safe)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum worktrees to consider (0 = all)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)

    try:
        plan = build_plan(
            skip_pr_check=args.skip_pr_check,
            limit=args.limit,
        )
    except Exception as exc:
        print(f"ERROR building reaper plan: {exc}", file=sys.stderr)
        return 1

    if args.apply:
        plan = apply_deletions(plan)

    if args.json:
        output = asdict(plan)
        print(json.dumps(output, indent=2, sort_keys=True))
    else:
        print(format_human(plan))

    return 0 if not plan.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
