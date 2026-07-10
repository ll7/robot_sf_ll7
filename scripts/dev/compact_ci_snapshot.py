#!/usr/bin/env python3
"""Emit a compact CI snapshot for PR queue monitoring.

This helper provides token-efficient CI state summaries for autonomous PR loops.
It avoids fetching full CI logs and instead reports compact check rollup state,
with optional drift sampling after timeout.

Use before delegating CI wait monitors or making merge-readiness decisions.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from scripts.dev._gh_pagination import is_likely_truncated

SCHEMA_VERSION = "compact_ci_snapshot.v1"
DEFAULT_REPO = "ll7/robot_sf_ll7"
DEFAULT_WORKFLOW = "CI"
DEFAULT_SAMPLE_LIMIT = 10
FAILURE_CONCLUSIONS = {
    "failure",
    "error",
    "cancelled",
    "timed_out",
    "action_required",
    "startup_failure",
}
PENDING_STATUSES = {"expected", "in_progress", "pending", "queued", "requested", "waiting"}


@dataclass(frozen=True, slots=True)
class CheckSummary:
    """Compact check rollup."""

    overall: str
    total: int
    by_conclusion: dict[str, int]
    by_status: dict[str, int]
    names: list[str]
    failed_names: list[str]
    pending_names: list[str]
    success_names: list[str]


@dataclass(frozen=True, slots=True)
class DriftSample:
    """CI drift sample after timeout."""

    source: str
    workflow: str
    sample_count: int
    median_seconds: int | None
    recommended_budget_seconds: int | None
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class PRSnapshot:
    """Compact PR CI snapshot."""

    number: int
    title: str
    state: str
    branch: str
    head_sha: str
    expected_head_sha: str | None
    head_matches_expected: bool | None
    mergeable: str
    checks: CheckSummary | None
    next_action: str
    freshness_key: str
    error: str | None


@dataclass(frozen=True, slots=True)
class SnapshotResult:
    """Full CI snapshot result."""

    schema: str
    repo: str
    prs: list[PRSnapshot]
    drift_sample: DriftSample | None
    generated_at_utc: str
    errors: list[str]


def _gh(args: list[str], *, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a GitHub CLI command."""
    try:
        return subprocess.run(
            ["gh", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            args=["gh", *args],
            returncode=124,
            stdout="",
            stderr=f"gh command timed out after {timeout} seconds",
        )
    except FileNotFoundError:
        return subprocess.CompletedProcess(
            args=["gh", *args],
            returncode=127,
            stdout="",
            stderr="gh CLI not found",
        )


def _rollup_conclusion(check: dict[str, Any]) -> str:
    """Normalize check conclusion."""
    conclusion = check.get("conclusion")
    if conclusion:
        return str(conclusion).lower()
    state = check.get("state")
    if state:
        return str(state).lower()
    return "pending"


def _rollup_status(check: dict[str, Any]) -> str:
    """Normalize check status."""
    status = check.get("status")
    if status:
        return str(status).lower()
    state = check.get("state")
    if not state:
        return "completed"
    state_str = str(state).lower()
    if state_str in {"success", "failure", "error"}:
        return "completed"
    return state_str


def _check_name(check: dict[str, Any]) -> str:
    """Return compact check name."""
    return str(check.get("name") or check.get("context") or "unknown")


def _build_check_summary(rollup: list[dict[str, Any]]) -> CheckSummary:
    """Build compact check summary from rollup."""
    valid_checks = [c for c in rollup if isinstance(c, dict)]
    conclusions: dict[str, int] = {}
    statuses: dict[str, int] = {}
    names: set[str] = set()
    failed_names: set[str] = set()
    pending_names: set[str] = set()
    success_names: set[str] = set()

    for check in valid_checks:
        conclusion = _rollup_conclusion(check)
        status = _rollup_status(check)
        name = _check_name(check)
        conclusions[conclusion] = conclusions.get(conclusion, 0) + 1
        statuses[status] = statuses.get(status, 0) + 1
        names.add(name)
        if conclusion in FAILURE_CONCLUSIONS:
            failed_names.add(name)
        elif status in PENDING_STATUSES:
            pending_names.add(name)
        elif conclusion in {"success", "neutral", "skipped"}:
            success_names.add(name)

    failure_count = sum(conclusions.get(c, 0) for c in FAILURE_CONCLUSIONS)
    pending_count = sum(statuses.get(s, 0) for s in PENDING_STATUSES)

    if failure_count:
        overall = "failure"
    elif pending_count or not valid_checks:
        overall = "pending"
    else:
        overall = "success"

    return CheckSummary(
        overall=overall,
        total=len(valid_checks),
        by_conclusion=dict(sorted(conclusions.items())),
        by_status=dict(sorted(statuses.items())),
        names=sorted(names),
        failed_names=sorted(failed_names),
        pending_names=sorted(pending_names),
        success_names=sorted(success_names),
    )


def _next_action(checks: CheckSummary | None, head_matches_expected: bool | None) -> str:
    """Return a compact next-action hint for the parent orchestrator."""
    if head_matches_expected is False:
        return "refresh_snapshot_expected_head_changed"
    if checks is None:
        return "wait_for_checks_or_verify_pr_head"
    if checks.overall == "failure":
        return "inspect_failed_check_excerpt"
    if checks.overall == "pending":
        return "await_ci_with_compact_monitor"
    return "review_merge_readiness"


def _fetch_pr_snapshot(
    number: int,
    repo: str,
    *,
    expected_head_sha: str | None = None,
) -> PRSnapshot:
    """Fetch compact PR CI snapshot."""
    result = _gh(
        [
            "pr",
            "view",
            str(number),
            "--repo",
            repo,
            "--json",
            "number,title,state,mergeable,headRefName,headRefOid,statusCheckRollup",
        ]
    )

    if result.returncode != 0:
        return PRSnapshot(
            number=number,
            title="",
            state="",
            branch="",
            head_sha="",
            expected_head_sha=expected_head_sha,
            head_matches_expected=None,
            mergeable="",
            checks=None,
            next_action="fix_pr_snapshot_fetch",
            freshness_key=f"pr-{number}:unavailable",
            error=result.stderr.strip() or f"gh returned exit code {result.returncode}",
        )

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return PRSnapshot(
            number=number,
            title="",
            state="",
            branch="",
            head_sha="",
            expected_head_sha=expected_head_sha,
            head_matches_expected=None,
            mergeable="",
            checks=None,
            next_action="fix_pr_snapshot_parse",
            freshness_key=f"pr-{number}:invalid-json",
            error=f"invalid JSON: {exc}",
        )

    rollup = data.get("statusCheckRollup", []) or []
    checks = _build_check_summary(rollup) if rollup else None
    head_sha = str(data.get("headRefOid", ""))
    head_matches_expected = expected_head_sha == head_sha if expected_head_sha else None

    return PRSnapshot(
        number=data.get("number", number),
        title=data.get("title", ""),
        state=data.get("state", ""),
        branch=data.get("headRefName", ""),
        head_sha=head_sha,
        expected_head_sha=expected_head_sha,
        head_matches_expected=head_matches_expected,
        mergeable=data.get("mergeable", ""),
        checks=checks,
        next_action=_next_action(checks, head_matches_expected),
        freshness_key=f"pr-{data.get('number', number)}:{head_sha}",
        error=None,
    )


def _parse_timestamp(value: Any) -> datetime | None:
    """Parse GitHub timestamp."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _duration_seconds(start: Any, end: Any) -> int | None:
    """Calculate elapsed seconds."""
    started = _parse_timestamp(start)
    completed = _parse_timestamp(end)
    if started is None or completed is None:
        return None
    try:
        return max(int((completed - started).total_seconds()), 0)
    except TypeError:
        return None


def _fetch_drift_sample(
    workflow: str,
    sample_limit: int,
    repo: str,
) -> DriftSample:
    """Fetch recent successful CI durations for drift analysis."""
    result = _gh(
        [
            "run",
            "list",
            "--workflow",
            workflow,
            "--repo",
            repo,
            "--status",
            "success",
            "--limit",
            str(sample_limit),
            "--json",
            "databaseId,displayTitle,status,conclusion,createdAt,updatedAt",
        ]
    )

    if result.returncode != 0:
        return DriftSample(
            source=f"error: {result.stderr.strip() or 'gh failed'}",
            workflow=workflow,
            sample_count=0,
            median_seconds=None,
            recommended_budget_seconds=None,
        )

    try:
        runs = json.loads(result.stdout)
    except json.JSONDecodeError:
        return DriftSample(
            source="gh run list",
            workflow=workflow,
            sample_count=0,
            median_seconds=None,
            recommended_budget_seconds=None,
        )

    if not isinstance(runs, list):
        return DriftSample(
            source="gh run list",
            workflow=workflow,
            sample_count=0,
            median_seconds=None,
            recommended_budget_seconds=None,
        )

    # Sampling call: a raw result at the cap means the drift window was capped, so
    # record it as a structured marker rather than treat the sample as exhaustive
    # (issue #5048 / #4991).
    truncated = is_likely_truncated(len(runs), limit=sample_limit)

    durations: list[int] = []
    for run in runs:
        if not isinstance(run, dict):
            continue
        if str(run.get("conclusion", "")).lower() != "success":
            continue
        duration = _duration_seconds(run.get("createdAt"), run.get("updatedAt"))
        if duration is not None:
            durations.append(duration)

    if not durations:
        return DriftSample(
            source="gh run list",
            workflow=workflow,
            sample_count=0,
            median_seconds=None,
            recommended_budget_seconds=None,
            truncated=truncated,
        )

    median_seconds = math.ceil(statistics.median(durations))
    return DriftSample(
        source="gh run list",
        workflow=workflow,
        sample_count=len(durations),
        median_seconds=median_seconds,
        recommended_budget_seconds=math.ceil(median_seconds * 1.3),
        truncated=truncated,
    )


def _now_utc() -> str:
    """Return ISO-8601 UTC timestamp."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def build_snapshot(
    pr_numbers: list[int],
    *,
    repo: str = DEFAULT_REPO,
    expected_head_sha: str | None = None,
    include_drift: bool = False,
    workflow: str = DEFAULT_WORKFLOW,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
) -> SnapshotResult:
    """Build compact CI snapshot."""
    errors: list[str] = []
    prs: list[PRSnapshot] = []

    for number in pr_numbers:
        pr = _fetch_pr_snapshot(number, repo=repo, expected_head_sha=expected_head_sha)
        if pr.error:
            errors.append(f"PR {number}: {pr.error}")
        prs.append(pr)

    drift_sample: DriftSample | None = None
    if include_drift:
        drift_sample = _fetch_drift_sample(
            workflow=workflow,
            sample_limit=sample_limit,
            repo=repo,
        )

    return SnapshotResult(
        schema=SCHEMA_VERSION,
        repo=repo,
        prs=prs,
        drift_sample=drift_sample,
        generated_at_utc=_now_utc(),
        errors=errors,
    )


def format_human(result: SnapshotResult) -> str:
    """Format as human-readable text."""
    lines = [
        f"CI Snapshot (schema: {result.schema})",
        f"  Repo: {result.repo}",
        f"  Generated: {result.generated_at_utc}",
        f"  PRs: {len(result.prs)}",
    ]

    for pr in result.prs:
        status = pr.checks.overall if pr.checks else "unknown"
        lines.append(f"    - PR #{pr.number}: {pr.title[:50]}... [{status}]")
        lines.append(f"        freshness: {pr.freshness_key}; next: {pr.next_action}")
        if pr.checks:
            lines.append(
                f"        checks: {pr.checks.total} total, "
                f"conclusions={pr.checks.by_conclusion}, "
                f"status={pr.checks.by_status}"
            )
            if pr.checks.failed_names:
                lines.append(f"        failed: {', '.join(pr.checks.failed_names)}")
            if pr.checks.pending_names:
                lines.append(f"        pending: {', '.join(pr.checks.pending_names)}")
        if pr.error:
            lines.append(f"        error: {pr.error}")

    if result.drift_sample:
        ds = result.drift_sample
        lines.append(
            f"  Drift sample: {ds.sample_count} runs, "
            f"median={ds.median_seconds}s, "
            f"recommended={ds.recommended_budget_seconds}s"
        )

    if result.errors:
        lines.append("  Errors:")
        for err in result.errors:
            lines.append(f"    - {err}")

    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "prs",
        nargs="+",
        type=int,
        help="PR numbers to snapshot",
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help="GitHub repo",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON",
    )
    parser.add_argument(
        "--expected-head-sha",
        help="Expected PR head SHA freshness key. Mark stale if the live PR head differs.",
    )
    parser.add_argument(
        "--include-drift",
        action="store_true",
        help="Include drift sample from recent runs",
    )
    parser.add_argument(
        "--workflow",
        default=DEFAULT_WORKFLOW,
        help="Workflow name for drift sampling",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=DEFAULT_SAMPLE_LIMIT,
        help="Recent runs to sample for drift",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)

    try:
        result = build_snapshot(
            pr_numbers=args.prs,
            repo=args.repo,
            expected_head_sha=args.expected_head_sha,
            include_drift=args.include_drift,
            workflow=args.workflow,
            sample_limit=args.sample_limit,
        )
    except Exception as exc:
        print(f"ERROR building CI snapshot: {exc}", file=sys.stderr)
        return 1

    if args.json:
        output = asdict(result)
        print(json.dumps(output, indent=2, sort_keys=True))
    else:
        print(format_human(result))

    return 0 if not result.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
