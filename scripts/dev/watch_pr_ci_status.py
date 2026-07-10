#!/usr/bin/env python3
"""Watch PR CI status with a stable default wait budget.

The normal path intentionally does not sample recent CI timings.  Runtime sampling is reserved for
the drift path after the default budget is exhausted while checks are still pending.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.dev._gh_pagination import is_likely_truncated  # noqa: E402
from scripts.dev.check_pr_ci_status import _fetch_ci_status  # noqa: E402

logger = logging.getLogger(__name__)

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
    budget_overridden: bool
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


def _parse_timestamp(value: Any) -> datetime | None:
    """Parse a GitHub timestamp."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _duration_seconds(start: Any, end: Any) -> int | None:
    """Return elapsed whole seconds for a GitHub run timestamp pair."""
    started = _parse_timestamp(start)
    completed = _parse_timestamp(end)
    if started is None or completed is None:
        return None
    try:
        return max(int((completed - started).total_seconds()), 0)
    except TypeError:
        return None


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
    if is_likely_truncated(len(runs), limit=limit):
        # Sampling call: hitting the cap is expected, but record a structured,
        # greppable marker so a capped drift window is never mistaken for the
        # full recent-run history (issue #5048 / #4991).
        logger.warning(
            "gh run list truncated: got %d rows at --limit %d for workflow %r; "
            "drift sample is capped, raise --sample-limit for a wider window",
            len(runs),
            limit,
            workflow,
        )
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


def watch_pr_ci_status(  # noqa: PLR0913, C901 - CLI/test seam with explicit injectable dependencies.
    *,
    pr_number: str,
    expected_head_sha: str = "",
    baseline_seconds: int = DEFAULT_BASELINE_SECONDS,
    multiplier: float = DEFAULT_MULTIPLIER,
    budget_override_seconds: int | None = None,
    poll_interval_seconds: int = DEFAULT_POLL_INTERVAL_SECONDS,
    workflow: str = DEFAULT_WORKFLOW,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
    fetch_status: Callable[..., dict[str, Any]] = _fetch_ci_status,
    fetch_durations: Callable[..., list[int]] = fetch_recent_successful_ci_durations,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    once: bool = False,
    emit_progress_json_every: int = 0,
    progress_stream: Any = sys.stderr,
) -> WatchResult:
    """Poll PR CI status until success, failure, stale head, or budget timeout."""
    if budget_override_seconds is not None:
        budget_seconds = max(budget_override_seconds, 0)
    else:
        budget_seconds = wait_budget_seconds(baseline_seconds, multiplier)
    deadline = monotonic() + budget_seconds
    last_status: dict[str, Any] = {}
    last_progress_at = 0.0
    poll_count = 0

    while True:
        last_status = fetch_status(pr_number)
        poll_count += 1
        head_sha = str(last_status.get("head_sha") or "")
        if last_status.get("status") == "error":
            return WatchResult(
                pr=pr_number,
                head_sha=head_sha,
                expected_head_sha=expected_head_sha,
                baseline_seconds=baseline_seconds,
                multiplier=multiplier,
                budget_seconds=budget_seconds,
                budget_overridden=budget_override_seconds is not None,
                poll_interval_seconds=poll_interval_seconds,
                final_status="error",
                checks={},
                error=str(last_status.get("error") or "unknown status error"),
                drift_sample=None,
            )
        state = str(last_status.get("state") or "").upper()
        if state in {"CLOSED", "MERGED"}:
            return WatchResult(
                pr=last_status.get("pr", pr_number),
                head_sha=head_sha,
                expected_head_sha=expected_head_sha,
                baseline_seconds=baseline_seconds,
                multiplier=multiplier,
                budget_seconds=budget_seconds,
                budget_overridden=budget_override_seconds is not None,
                poll_interval_seconds=poll_interval_seconds,
                final_status="error",
                checks=last_status.get("checks", {}),
                error=f"PR is in terminal state: {state}",
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
                budget_overridden=budget_override_seconds is not None,
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
                budget_overridden=budget_override_seconds is not None,
                poll_interval_seconds=poll_interval_seconds,
                final_status=str(overall),
                checks=checks,
                error="",
                drift_sample=None,
            )
        if once:
            return WatchResult(
                pr=last_status.get("pr", pr_number),
                head_sha=head_sha,
                expected_head_sha=expected_head_sha,
                baseline_seconds=baseline_seconds,
                multiplier=multiplier,
                budget_seconds=budget_seconds,
                budget_overridden=budget_override_seconds is not None,
                poll_interval_seconds=poll_interval_seconds,
                final_status=str(overall or "pending"),
                checks=checks,
                error="",
                drift_sample=None,
            )

        remaining = deadline - monotonic()
        if emit_progress_json_every > 0:
            now = monotonic()
            if last_progress_at <= 0 or now - last_progress_at >= emit_progress_json_every:
                print(
                    json.dumps(
                        {
                            "schema": "pr_ci_watch_progress.v1",
                            "pr": last_status.get("pr", pr_number),
                            "head_sha": head_sha,
                            "expected_head_sha": expected_head_sha,
                            "poll_count": poll_count,
                            "status": str(overall or "pending"),
                            "remaining_seconds": max(int(remaining), 0),
                            "checks": checks,
                        },
                        sort_keys=True,
                    ),
                    file=progress_stream,
                    flush=True,
                )
                last_progress_at = now
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
                budget_overridden=budget_override_seconds is not None,
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
    ]
    if result.budget_overridden:
        lines.append(f"  budget: {result.budget_seconds}s (direct override)")
    else:
        lines.append(
            f"  budget: {result.budget_seconds}s "
            f"(baseline {result.baseline_seconds}s * {result.multiplier:g})"
        )
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


EXAMPLE = """\
long-poll with SHA guard (agents should always pass --expected-head-sha):

  uv run python scripts/dev/watch_pr_ci_status.py 123 --json \\
      --expected-head-sha $(gh pr view 123 --json headRefOid -q .headRefOid) \\
      --poll-interval 90 --budget-seconds 900
"""


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=EXAMPLE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
        "--poll-interval",
        type=int,
        default=DEFAULT_POLL_INTERVAL_SECONDS,
        help="Seconds between CI status polls.",
    )
    parser.add_argument(
        "--budget-seconds",
        type=int,
        default=None,
        help=(
            "Override the computed wait budget (baseline * multiplier) with a fixed "
            "second count.  Use this when agents must not inherit drift-tuned budgets."
        ),
    )
    parser.add_argument("--sample-limit", type=int, default=DEFAULT_SAMPLE_LIMIT)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Fetch one current CI status snapshot without waiting for checks to finish.",
    )
    parser.add_argument(
        "--emit-progress-json-every",
        type=int,
        default=0,
        help="Emit compact progress JSON to stderr at this interval while waiting.",
    )
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
            budget_override_seconds=args.budget_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
            workflow=args.workflow,
            sample_limit=args.sample_limit,
            fetch_status=_fetch_ci_status,
            fetch_durations=fetch_recent_successful_ci_durations,
            once=args.once,
            emit_progress_json_every=args.emit_progress_json_every,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(f"ERROR watching PR CI: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"ERROR: Invalid argument: {exc}", file=sys.stderr)
        return 1
    print(result.to_json() if args.json else format_human(result))
    if result.final_status == "success":
        return 0
    if result.final_status in {"timeout", "pending"}:
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
