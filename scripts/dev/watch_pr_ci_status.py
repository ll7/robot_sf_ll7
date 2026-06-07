#!/usr/bin/env python3
"""Watch PR CI status with a stable default wait budget.

The normal path intentionally does not sample recent CI timings.  Runtime sampling is reserved for
the drift path after the default budget is exhausted while checks are still pending.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from scripts.dev.check_pr_ci_status import _fetch_ci_status

DEFAULT_BASELINE_SECONDS = 920
DEFAULT_MULTIPLIER = 1.3
DEFAULT_POLL_INTERVAL_SECONDS = 120
DEFAULT_WORKFLOW = "CI"
DEFAULT_SAMPLE_LIMIT = 10


@dataclass(frozen=True, slots=True)
class DriftSample:
    """Recent successful CI timing sample collected after a timeout."""

    source: str
    workflow: str
    sample_count: int
    median_seconds: int | None
    recommended_budget_seconds: int | None


@dataclass(frozen=True, slots=True)
class WatchResult:
    """Final PR CI watch result."""

    pr: int | str
    head_sha: str
    expected_head_sha: str
    baseline_seconds: int
    multiplier: float
    budget_seconds: int
    poll_interval_seconds: int
    final_status: str
    checks: dict[str, Any]
    error: str
    drift_sample: DriftSample | None

    def to_json(self) -> str:
        """Serialize the result as deterministic JSON."""
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def wait_budget_seconds(baseline_seconds: int, multiplier: float) -> int:
    """Return the rounded-up CI wait budget."""
    if baseline_seconds < 0:
        raise ValueError("baseline_seconds must be non-negative")
    if multiplier <= 0:
        raise ValueError("multiplier must be positive")
    return math.ceil(baseline_seconds * multiplier)


def _parse_timestamp(value: str | None) -> datetime | None:
    """Parse a GitHub timestamp."""
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _duration_seconds(start: str | None, end: str | None) -> int | None:
    """Return elapsed whole seconds for a GitHub run timestamp pair."""
    started = _parse_timestamp(start)
    completed = _parse_timestamp(end)
    if started is None or completed is None:
        return None
    return max(int((completed - started).total_seconds()), 0)


def _gh(args: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a GitHub CLI command."""
    return subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def fetch_recent_successful_ci_durations(
    *,
    workflow: str = DEFAULT_WORKFLOW,
    limit: int = DEFAULT_SAMPLE_LIMIT,
) -> list[int]:
    """Fetch recent successful workflow durations from `gh run list`."""
    result = _gh(
        [
            "run",
            "list",
            "--workflow",
            workflow,
            "--status",
            "success",
            "--limit",
            str(limit),
            "--json",
            "databaseId,displayTitle,status,conclusion,createdAt,updatedAt",
        ]
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"gh returned exit code {result.returncode}")
    try:
        runs = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse gh run list JSON: {exc}") from exc
    if not isinstance(runs, list):
        raise RuntimeError("gh run list output is not a JSON array")
    durations: list[int] = []
    for run in runs:
        if not isinstance(run, dict):
            continue
        if str(run.get("conclusion", "")).lower() != "success":
            continue
        duration = _duration_seconds(run.get("createdAt"), run.get("updatedAt"))
        if duration is not None:
            durations.append(duration)
    return durations


def _build_drift_sample(
    *,
    workflow: str,
    sample_limit: int,
    multiplier: float,
    fetch_durations: Callable[..., list[int]],
) -> DriftSample:
    """Collect optional drift evidence after the default budget is exhausted."""
    try:
        durations = fetch_durations(workflow=workflow, limit=sample_limit)
    except Exception as exc:
        return DriftSample(
            source=f"error: {exc}",
            workflow=workflow,
            sample_count=0,
            median_seconds=None,
            recommended_budget_seconds=None,
        )
    if not durations:
        return DriftSample(
            source="gh run list",
            workflow=workflow,
            sample_count=0,
            median_seconds=None,
            recommended_budget_seconds=None,
        )
    median_seconds = math.ceil(statistics.median(durations))
    return DriftSample(
        source="gh run list",
        workflow=workflow,
        sample_count=len(durations),
        median_seconds=median_seconds,
        recommended_budget_seconds=wait_budget_seconds(median_seconds, multiplier),
    )


def watch_pr_ci_status(  # noqa: PLR0913 - CLI/test seam with explicit injectable dependencies.
    *,
    pr_number: str,
    expected_head_sha: str = "",
    baseline_seconds: int = DEFAULT_BASELINE_SECONDS,
    multiplier: float = DEFAULT_MULTIPLIER,
    poll_interval_seconds: int = DEFAULT_POLL_INTERVAL_SECONDS,
    workflow: str = DEFAULT_WORKFLOW,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
    fetch_status: Callable[..., dict[str, Any]] = _fetch_ci_status,
    fetch_durations: Callable[..., list[int]] = fetch_recent_successful_ci_durations,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> WatchResult:
    """Poll PR CI status until success, failure, stale head, or budget timeout."""
    budget_seconds = wait_budget_seconds(baseline_seconds, multiplier)
    deadline = monotonic() + budget_seconds
    last_status: dict[str, Any] = {}

    while True:
        last_status = fetch_status(pr_number)
        head_sha = str(last_status.get("head_sha") or "")
        if last_status.get("status") == "error":
            return WatchResult(
                pr=pr_number,
                head_sha=head_sha,
                expected_head_sha=expected_head_sha,
                baseline_seconds=baseline_seconds,
                multiplier=multiplier,
                budget_seconds=budget_seconds,
                poll_interval_seconds=poll_interval_seconds,
                final_status="error",
                checks={},
                error=str(last_status.get("error") or "unknown status error"),
                drift_sample=None,
            )
        if expected_head_sha and head_sha and head_sha != expected_head_sha:
            return WatchResult(
                pr=last_status.get("pr", pr_number),
                head_sha=head_sha,
                expected_head_sha=expected_head_sha,
                baseline_seconds=baseline_seconds,
                multiplier=multiplier,
                budget_seconds=budget_seconds,
                poll_interval_seconds=poll_interval_seconds,
                final_status="error",
                checks=last_status.get("checks", {}),
                error="PR head SHA changed while waiting for CI",
                drift_sample=None,
            )

        checks = last_status.get("checks", {})
        overall = checks.get("overall")
        if overall in {"success", "failure"}:
            return WatchResult(
                pr=last_status.get("pr", pr_number),
                head_sha=head_sha,
                expected_head_sha=expected_head_sha,
                baseline_seconds=baseline_seconds,
                multiplier=multiplier,
                budget_seconds=budget_seconds,
                poll_interval_seconds=poll_interval_seconds,
                final_status=str(overall),
                checks=checks,
                error="",
                drift_sample=None,
            )

        remaining = deadline - monotonic()
        if remaining <= 0:
            drift_sample = _build_drift_sample(
                workflow=workflow,
                sample_limit=sample_limit,
                multiplier=multiplier,
                fetch_durations=fetch_durations,
            )
            return WatchResult(
                pr=last_status.get("pr", pr_number),
                head_sha=head_sha,
                expected_head_sha=expected_head_sha,
                baseline_seconds=baseline_seconds,
                multiplier=multiplier,
                budget_seconds=budget_seconds,
                poll_interval_seconds=poll_interval_seconds,
                final_status="timeout",
                checks=checks,
                error="CI remained pending after wait budget",
                drift_sample=drift_sample,
            )
        sleep(max(min(float(poll_interval_seconds), remaining), 0.0))


def format_human(result: WatchResult) -> str:
    """Format a compact human-readable monitor summary."""
    lines = [
        f"PR #{result.pr} CI watch: {result.final_status}",
        (f"  head: {result.head_sha}  |  expected: {result.expected_head_sha or 'not set'}"),
        (
            f"  budget: {result.budget_seconds}s "
            f"(baseline {result.baseline_seconds}s * {result.multiplier:g})"
        ),
    ]
    if result.checks:
        lines.append(f"  checks: {result.checks.get('overall', 'unknown')}")
    if result.error:
        lines.append(f"  error: {result.error}")
    if result.drift_sample is not None:
        sample = result.drift_sample
        lines.append(
            "  drift_sample: "
            f"{sample.sample_count} runs, median={sample.median_seconds}, "
            f"recommended_budget={sample.recommended_budget_seconds}, source={sample.source}"
        )
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pr_number", help="GitHub PR number to monitor.")
    parser.add_argument("--expected-head-sha", default="", help="Optional PR head SHA guard.")
    parser.add_argument(
        "--workflow", default=DEFAULT_WORKFLOW, help="Workflow name for drift sampling."
    )
    parser.add_argument(
        "--baseline-seconds",
        type=int,
        default=DEFAULT_BASELINE_SECONDS,
        help="Default CI runtime baseline; recent runs are sampled only after timeout.",
    )
    parser.add_argument("--multiplier", type=float, default=DEFAULT_MULTIPLIER)
    parser.add_argument(
        "--poll-interval-seconds",
        type=int,
        default=DEFAULT_POLL_INTERVAL_SECONDS,
    )
    parser.add_argument("--sample-limit", type=int, default=DEFAULT_SAMPLE_LIMIT)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    try:
        result = watch_pr_ci_status(
            pr_number=args.pr_number,
            expected_head_sha=args.expected_head_sha,
            baseline_seconds=args.baseline_seconds,
            multiplier=args.multiplier,
            poll_interval_seconds=args.poll_interval_seconds,
            workflow=args.workflow,
            sample_limit=args.sample_limit,
            fetch_status=_fetch_ci_status,
            fetch_durations=fetch_recent_successful_ci_durations,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(f"ERROR watching PR CI: {exc}", file=sys.stderr)
        return 1
    print(result.to_json() if args.json else format_human(result))
    if result.final_status == "success":
        return 0
    if result.final_status == "timeout":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
