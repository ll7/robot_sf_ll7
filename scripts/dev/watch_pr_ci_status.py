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
from scripts.dev.check_pr_ci_status import (  # noqa: E402
    PENDING_STATUSES,
    _actions_run_id,
    _enrich_rest_check_runs,
    _fetch_ci_status,
    _latest_check_runs_with_evidence,
    _rest_api_get,
    _rest_check_runs_to_rollup,
    _summarize_check_runs,
)

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
    target_kind: str = "pr_head"
    target_sha: str = ""

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


def _fetch_exact_commit_check_runs(commit_sha: str) -> Any:
    """Fetch check runs for one exact commit through the repository REST endpoint."""
    return _rest_api_get(f"commits/{commit_sha}/check-runs?per_page=100")


def _fetch_exact_commit_workflow_runs(commit_sha: str) -> Any:
    """Fetch Actions workflow runs for one exact commit through the REST endpoint."""
    return _rest_api_get(f"actions/runs?head_sha={commit_sha}&per_page=100")


def _workflow_run_id(run: dict[str, Any]) -> int | None:
    """Return a positive Actions workflow-run ID, or ``None`` for malformed metadata."""
    raw_id = run.get("id")
    if isinstance(raw_id, bool):
        return None
    if isinstance(raw_id, int):
        run_id = raw_id
    elif isinstance(raw_id, str) and raw_id.isdigit():
        run_id = int(raw_id)
    else:
        return None
    return run_id if run_id > 0 else None


def _workflow_run_workflow_id(run: dict[str, Any]) -> str:
    """Return the stable workflow identity exposed by Actions run metadata."""
    for key in ("workflow_id", "workflowId"):
        value = run.get(key)
        if value is None:
            continue
        identity = str(value).strip()
        if identity and identity != "0":
            return identity
    return ""


def _workflow_run_started_at(run: dict[str, Any]) -> str:
    """Return the best timestamp for ordering a replacement workflow run."""
    for key in ("run_started_at", "runStartedAt", "created_at", "createdAt"):
        value = run.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _exact_workflow_runs(payload: Any, commit_sha: str) -> list[dict[str, Any]]:
    """Keep only well-formed workflow runs that independently confirm the exact SHA."""
    if not isinstance(payload, dict):
        return []
    workflow_runs = payload.get("workflow_runs")
    if not isinstance(workflow_runs, list):
        return []
    return [
        run
        for run in workflow_runs
        if isinstance(run, dict) and str(run.get("head_sha") or "") == commit_sha
    ]


def _bind_workflow_run_identities(
    check_runs: list[dict[str, Any]], workflow_runs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Use exact-SHA workflow metadata to fill missing check-run workflow identities."""
    workflow_ids_by_run_id = {
        str(run_id): _workflow_run_workflow_id(run)
        for run in workflow_runs
        if (run_id := _workflow_run_id(run)) is not None and _workflow_run_workflow_id(run)
    }
    bound_runs: list[dict[str, Any]] = []
    for check_run in check_runs:
        bound = dict(check_run)
        details_url = str(bound.get("details_url") or bound.get("detailsUrl") or "")
        run_id = _actions_run_id(details_url)
        workflow_id = workflow_ids_by_run_id.get(str(run_id)) if run_id is not None else None
        if workflow_id and not (bound.get("workflow_id") or bound.get("workflowId")):
            bound["workflow_id"] = workflow_id
        bound_runs.append(bound)
    return bound_runs


def _newer_workflow_run(
    check: dict[str, Any], workflow_runs: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Return the newest exact-SHA workflow run replacing one cancelled check."""
    details_url = str(check.get("detailsUrl") or check.get("details_url") or "")
    old_run_id = _actions_run_id(details_url)
    workflow_id = str(check.get("workflowId") or check.get("workflow_id") or "")
    if old_run_id is None or not workflow_id:
        return None
    candidates = [
        run
        for run in workflow_runs
        if _workflow_run_id(run) is not None
        and _workflow_run_id(run) > old_run_id
        and _workflow_run_workflow_id(run) == workflow_id
    ]
    if not candidates:
        return None
    return max(
        candidates, key=lambda run: (_workflow_run_id(run) or 0, _workflow_run_started_at(run))
    )


def _has_materialized_replacement(
    check: dict[str, Any], replacement: dict[str, Any], visible_checks: list[dict[str, Any]]
) -> bool:
    """Return whether the replacement workflow already exposed this named job check."""
    replacement_run_id = _workflow_run_id(replacement)
    if replacement_run_id is None:
        return False
    check_name = str(check.get("name") or check.get("context") or "")
    if not check_name:
        return False
    for visible in visible_checks:
        visible_name = str(visible.get("name") or visible.get("context") or "")
        details_url = str(visible.get("details_url") or visible.get("detailsUrl") or "")
        if visible_name == check_name and _actions_run_id(details_url) == replacement_run_id:
            return True
    return False


def _replacement_check_shape(
    check: dict[str, Any], replacement: dict[str, Any], commit_sha: str
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Build one representative check for a replacement job not yet materialized by GitHub."""
    replacement_run_id = _workflow_run_id(replacement)
    workflow_id = _workflow_run_workflow_id(replacement)
    started_at = _workflow_run_started_at(replacement)
    if replacement_run_id is None or not workflow_id or not started_at:
        return None

    raw_status = str(replacement.get("status") or "").lower()
    raw_conclusion = str(replacement.get("conclusion") or "").lower()
    # A workflow-level terminal conclusion does not prove that this individual job's
    # check-run has materialized. Keep the representative pending until the job check
    # itself is visible; the conclusion remains diagnostic evidence in the marker.
    check_status = raw_status if raw_status in PENDING_STATUSES else "in_progress"
    check_conclusion: str | None = None

    replacement_url = str(
        replacement.get("html_url") or replacement.get("htmlUrl") or replacement.get("url") or ""
    )
    materialization = {
        "source": "actions_workflow_run_metadata",
        "replacement_run_id": replacement_run_id,
        "replacement_run_url": replacement_url or None,
        "workflow_id": workflow_id,
        "run_status": raw_status or None,
        "run_conclusion": raw_conclusion or None,
        "check_status": check_status,
        "check_conclusion": check_conclusion,
    }
    synthetic = {
        "__typename": "CheckRun",
        "name": str(check.get("name") or check.get("context") or "unknown"),
        "status": check_status,
        "conclusion": check_conclusion,
        "head_sha": commit_sha,
        "started_at": started_at,
        "completed_at": None,
        "details_url": replacement_url,
        "workflow_id": workflow_id,
        "__replacement_materialization": materialization,
    }
    return synthetic, materialization


def _materialize_missing_replacements(
    check_runs: list[dict[str, Any]],
    *,
    commit_sha: str,
    fetch_workflow_runs: Callable[[str], Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Add fail-closed representatives for exact-SHA replacement jobs absent from REST checks."""
    normalized = _rest_check_runs_to_rollup(check_runs)
    effective, _, _ = _latest_check_runs_with_evidence(normalized)
    if not any(str(check.get("conclusion") or "").lower() == "cancelled" for check in effective):
        return check_runs, []

    try:
        workflow_runs = _exact_workflow_runs(fetch_workflow_runs(commit_sha), commit_sha)
    except (OSError, subprocess.SubprocessError, TypeError, ValueError):
        # A replacement lookup is optional evidence. Any lookup failure leaves the original
        # cancellation visible and therefore preserves fail-closed behavior.
        return check_runs, []
    if not workflow_runs:
        return check_runs, []

    bound_check_runs = _bind_workflow_run_identities(check_runs, workflow_runs)
    normalized = _rest_check_runs_to_rollup(bound_check_runs)
    effective, _, _ = _latest_check_runs_with_evidence(normalized)
    additions: list[dict[str, Any]] = []
    materializations: list[dict[str, Any]] = []
    for check in effective:
        if str(check.get("conclusion") or "").lower() != "cancelled":
            continue
        replacement = _newer_workflow_run(check, workflow_runs)
        if replacement is None or _has_materialized_replacement(
            check, replacement, bound_check_runs
        ):
            continue
        shape = _replacement_check_shape(check, replacement, commit_sha)
        if shape is None:
            continue
        synthetic, materialization = shape
        additions.append(synthetic)
        materializations.append(materialization)
    return bound_check_runs + additions, materializations


def fetch_exact_commit_ci_status(
    commit_sha: str,
    *,
    fetch_check_runs: Callable[[str], Any] = _fetch_exact_commit_check_runs,
    fetch_workflow_runs: Callable[[str], Any] = _fetch_exact_commit_workflow_runs,
) -> dict[str, Any]:
    """Return a fail-closed check summary for one exact commit SHA.

    The commit endpoint is intentionally independent of PR lifecycle state.  This supports the
    post-merge readback window where the PR is already terminal but its merge-commit checks are
    still running. If a visible cancelled check has a newer exact-SHA workflow run, Actions run
    metadata supplies a pending representative until that job's check record materializes.
    """
    commit_sha = commit_sha.strip()
    if not commit_sha:
        return {"status": "error", "error": "post-merge commit SHA is required"}
    payload = fetch_check_runs(commit_sha)
    if not isinstance(payload, dict):
        return {
            "status": "error",
            "head_sha": commit_sha,
            "error": f"could not fetch check runs for exact commit {commit_sha}",
        }
    check_runs = payload.get("check_runs")
    if not isinstance(check_runs, list):
        return {
            "status": "error",
            "head_sha": commit_sha,
            "error": f"check-run response for exact commit {commit_sha} was malformed",
        }
    mismatched_shas = sorted(
        {
            str(check_run.get("head_sha") or "")
            for check_run in check_runs
            if isinstance(check_run, dict)
            and check_run.get("head_sha")
            and str(check_run.get("head_sha")) != commit_sha
        }
    )
    if mismatched_shas:
        return {
            "status": "error",
            "head_sha": commit_sha,
            "error": (
                f"check-run response SHA mismatch for exact commit {commit_sha}: "
                + ", ".join(mismatched_shas)
            ),
        }
    raw_check_runs = [check_run for check_run in check_runs if isinstance(check_run, dict)]
    # The commit endpoint returns REST-shaped fields (for example ``workflow_id`` and
    # ``started_at``), while the shared rerun reducer consumes the normalized rollup shape.
    # Normalize and enrich before summarizing so an older cancelled run cannot override a newer
    # exact-commit replacement.  Unknown workflow identities remain independent and fail closed.
    enriched_check_runs = _enrich_rest_check_runs(raw_check_runs)
    enriched_check_runs, materializations = _materialize_missing_replacements(
        enriched_check_runs,
        commit_sha=commit_sha,
        fetch_workflow_runs=fetch_workflow_runs,
    )
    normalized_check_runs = _rest_check_runs_to_rollup(enriched_check_runs)
    checks, _ = _summarize_check_runs(normalized_check_runs)
    if materializations:
        checks["replacement_materialization"] = materializations
        if any(item["check_status"] in PENDING_STATUSES for item in materializations):
            checks["pending_reason"] = "replacement_check_materialization"
            checks["diagnostic"] = "workflow_replacement_check_materialization"
    return {
        "status": "ok",
        "head_sha": commit_sha,
        "commit_sha": commit_sha,
        "target_kind": "merge_commit",
        "checks": checks,
    }


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
    post_merge_commit_sha: str = "",
    fetch_commit_status: Callable[..., dict[str, Any]] = fetch_exact_commit_ci_status,
    fetch_durations: Callable[..., list[int]] = fetch_recent_successful_ci_durations,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    once: bool = False,
    emit_progress_json_every: int = 0,
    progress_stream: Any = sys.stderr,
) -> WatchResult:
    """Poll PR or exact merge-commit CI status until a terminal result or timeout."""
    expected_head_sha = expected_head_sha.strip()
    post_merge_commit_sha = post_merge_commit_sha.strip()
    if post_merge_commit_sha:
        if expected_head_sha and expected_head_sha != post_merge_commit_sha:
            raise ValueError(
                "--expected-head-sha must match --post-merge-commit-sha when both are provided"
            )
        expected_head_sha = post_merge_commit_sha
    target_kind = "merge_commit" if post_merge_commit_sha else "pr_head"
    target_sha = post_merge_commit_sha or expected_head_sha
    if budget_override_seconds is not None:
        budget_seconds = max(budget_override_seconds, 0)
    else:
        budget_seconds = wait_budget_seconds(baseline_seconds, multiplier)
    deadline = monotonic() + budget_seconds
    last_status: dict[str, Any] = {}
    last_progress_at = 0.0
    poll_count = 0

    while True:
        last_status = (
            fetch_commit_status(post_merge_commit_sha)
            if post_merge_commit_sha
            else fetch_status(pr_number)
        )
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
                target_kind=target_kind,
                target_sha=target_sha or head_sha,
            )
        state = str(last_status.get("state") or "").upper()
        if not post_merge_commit_sha and state in {"CLOSED", "MERGED"}:
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
                target_kind=target_kind,
                target_sha=target_sha or head_sha,
            )
        if post_merge_commit_sha and head_sha != expected_head_sha:
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
                error="merge commit SHA did not match the requested exact target",
                drift_sample=None,
                target_kind=target_kind,
                target_sha=target_sha or head_sha,
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
                error=(
                    "merge commit SHA did not match the requested exact target"
                    if post_merge_commit_sha
                    else "PR head SHA changed while waiting for CI"
                ),
                drift_sample=None,
                target_kind=target_kind,
                target_sha=target_sha or head_sha,
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
                target_kind=target_kind,
                target_sha=target_sha or head_sha,
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
                target_kind=target_kind,
                target_sha=target_sha or head_sha,
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
                target_kind=target_kind,
                target_sha=target_sha or head_sha,
            )
        sleep(max(min(float(poll_interval_seconds), remaining), 0.0))


def format_human(result: WatchResult) -> str:
    """Format a compact human-readable monitor summary."""
    if result.target_kind == "merge_commit":
        target_line = (
            f"  merge commit: {result.target_sha or result.head_sha}  |  "
            f"observed: {result.head_sha or 'not available'}  |  "
            f"expected: {result.expected_head_sha or 'not set'}"
        )
    else:
        target_line = (
            f"  head: {result.head_sha}  |  expected: {result.expected_head_sha or 'not set'}"
        )
    lines = [
        f"PR #{result.pr} CI watch: {result.final_status}",
        target_line,
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

post-merge exact commit readback:

  uv run python scripts/dev/watch_pr_ci_status.py 123 --json --post-merge-commit-sha "$MERGE_SHA"
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
        "--post-merge-commit-sha",
        default="",
        help=(
            "Opt in to exact check-run polling for a merged PR commit; this bypasses the "
            "terminal PR-state stop and requires the supplied SHA."
        ),
    )
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
            post_merge_commit_sha=args.post_merge_commit_sha,
            baseline_seconds=args.baseline_seconds,
            multiplier=args.multiplier,
            budget_override_seconds=args.budget_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
            workflow=args.workflow,
            sample_limit=args.sample_limit,
            fetch_status=_fetch_ci_status,
            fetch_commit_status=fetch_exact_commit_ci_status,
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
