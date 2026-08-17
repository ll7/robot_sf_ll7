#!/usr/bin/env python3
"""Reconcile a delayed GitHub Actions check-run for the reproducibility job.

GitHub Actions normally publishes a check-run whose identifier matches the
Actions job identifier.  A hosted-state race can leave that check-run pending
after the job completed successfully.  This helper is deliberately narrow:
it can only reconcile the exact successful ``reproducibility-check`` job for
the current workflow run, head SHA, and attempt.  Any mismatch or non-success
condition fails closed and is reported without mutating hosted state.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

SCHEMA = "reproducibility_check_run_reconciliation.v1"
DEFAULT_JOB_NAME = "reproducibility-check"
PENDING_CHECK_STATUSES = frozenset({"queued", "in_progress"})
Runner = Callable[[Sequence[str], str | None], Any]


class ReconciliationError(RuntimeError):
    """Raised when the exact hosted check-run cannot be verified safely."""


def _run_gh(args: Sequence[str], input_text: str | None = None) -> Any:
    """Run ``gh api`` and decode its JSON response."""

    result = subprocess.run(
        ["gh", *args],
        input=input_text,
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no response"
        raise ReconciliationError(f"GitHub API request failed: {detail}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ReconciliationError("GitHub API returned invalid JSON") from exc


class GitHubActionsClient:
    """Small injectable REST client used by the reconciler and its tests."""

    def __init__(self, runner: Runner = _run_gh) -> None:
        """Create a client backed by the supplied REST command runner."""

        self._runner = runner

    def list_jobs(self, repository: str, run_id: int) -> list[dict[str, Any]]:
        """Return the jobs recorded for one Actions workflow run."""

        payload = self._runner(
            ["api", f"repos/{repository}/actions/runs/{run_id}/jobs?per_page=100"],
            None,
        )
        jobs = payload.get("jobs") if isinstance(payload, dict) else None
        if not isinstance(jobs, list):
            raise ReconciliationError("Actions jobs response did not contain a jobs list")
        return [job for job in jobs if isinstance(job, dict)]

    def get_check_run(self, repository: str, check_run_id: int) -> dict[str, Any]:
        """Return one check-run by its Actions job/check-run identifier."""

        payload = self._runner(
            ["api", f"repos/{repository}/check-runs/{check_run_id}"],
            None,
        )
        if not isinstance(payload, dict):
            raise ReconciliationError("check-run response was not an object")
        return payload

    def complete_check_run(
        self,
        repository: str,
        check_run_id: int,
        *,
        output: dict[str, str],
    ) -> dict[str, Any]:
        """Complete a pending check-run with a success reconciliation payload."""

        payload = self._runner(
            [
                "api",
                "--method",
                "PATCH",
                f"repos/{repository}/check-runs/{check_run_id}",
                "--input",
                "-",
            ],
            json.dumps(
                {
                    "status": "completed",
                    "conclusion": "success",
                    "output": output,
                },
            ),
        )
        if not isinstance(payload, dict):
            raise ReconciliationError("check-run update response was not an object")
        return payload


def _as_int(value: Any, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ReconciliationError(f"{field} is not an integer: {value!r}") from exc


def _select_job(
    jobs: list[dict[str, Any]],
    *,
    repository: str,
    run_id: int,
    run_attempt: int,
    head_sha: str,
    job_name: str,
) -> dict[str, Any]:
    """Select exactly one job matching the current run identity."""

    matches = []
    for job in jobs:
        if job.get("name") != job_name:
            continue
        if job.get("run_id") is None or _as_int(job["run_id"], "job.run_id") != run_id:
            continue
        observed_attempt = job.get("run_attempt")
        if observed_attempt is None or _as_int(observed_attempt, "job.run_attempt") != run_attempt:
            continue
        if job.get("head_sha") != head_sha:
            continue
        matches.append(job)
    if len(matches) != 1:
        raise ReconciliationError(
            "expected exactly one matching reproducibility job "
            f"for {repository} run={run_id} attempt={run_attempt} head={head_sha}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _check_run_id_for_job(job: dict[str, Any], repository: str) -> int:
    """Extract and validate the check-run ID advertised by an Actions job."""

    check_run_url = job.get("check_run_url")
    if not isinstance(check_run_url, str):
        raise ReconciliationError("the matching Actions job did not advertise a check-run URL")
    parsed = urlparse(check_run_url)
    expected_prefix = f"/repos/{repository}/check-runs/"
    if (
        parsed.scheme != "https"
        or parsed.netloc != "api.github.com"
        or not parsed.path.startswith(expected_prefix)
        or parsed.query
        or parsed.fragment
    ):
        raise ReconciliationError("the matching Actions job advertised an unexpected check-run URL")
    check_run_id_text = parsed.path.removeprefix(expected_prefix)
    if not check_run_id_text.isdigit():
        raise ReconciliationError("the matching Actions job advertised a non-numeric check-run ID")
    return int(check_run_id_text)


def _check_identity(
    check_run: dict[str, Any],
    *,
    job_name: str,
    head_sha: str,
    check_run_id: int,
) -> None:
    """Reject a check-run that is not the Actions run we intend to repair."""

    if check_run.get("name") != job_name:
        raise ReconciliationError("matching job id did not resolve to the expected check-run name")
    if _as_int(check_run.get("id"), "check_run.id") != check_run_id:
        raise ReconciliationError("REST read-back returned a different check-run ID")
    if check_run.get("head_sha") != head_sha:
        raise ReconciliationError(
            "check-run head SHA does not match the current reproducibility job"
        )
    app = check_run.get("app")
    if not isinstance(app, dict) or app.get("slug") != "github-actions":
        raise ReconciliationError("refusing to mutate a non-GitHub-Actions check-run")


def reconcile_successful_job(
    client: GitHubActionsClient,
    *,
    repository: str,
    run_id: int,
    run_attempt: int,
    head_sha: str,
    job_name: str = DEFAULT_JOB_NAME,
) -> dict[str, Any]:
    """Reconcile one successful job and return a machine-readable result."""

    jobs = client.list_jobs(repository, run_id)
    job = _select_job(
        jobs,
        repository=repository,
        run_id=run_id,
        run_attempt=run_attempt,
        head_sha=head_sha,
        job_name=job_name,
    )
    if job.get("status") != "completed" or job.get("conclusion") != "success":
        raise ReconciliationError(
            "the matching Actions job is not completed successfully: "
            f"status={job.get('status')!r} conclusion={job.get('conclusion')!r}"
        )

    job_id = _as_int(job.get("id"), "job.id")
    check_run_id = _check_run_id_for_job(job, repository)
    check_run = client.get_check_run(repository, check_run_id)
    _check_identity(
        check_run,
        job_name=job_name,
        head_sha=head_sha,
        check_run_id=check_run_id,
    )
    initial_status = check_run.get("status")
    initial_conclusion = check_run.get("conclusion")

    base = {
        "schema": SCHEMA,
        "repository": repository,
        "run_id": run_id,
        "run_attempt": run_attempt,
        "head_sha": head_sha,
        "job_name": job_name,
        "job_id": job_id,
        "check_run_id": check_run_id,
        "job": {"status": job.get("status"), "conclusion": job.get("conclusion")},
        "check_run": {"status": initial_status, "conclusion": initial_conclusion},
        "fail_closed": True,
    }

    if initial_status == "completed" and initial_conclusion == "success":
        return {**base, "status": "healthy", "action": "none"}
    if initial_status == "completed":
        raise ReconciliationError(
            "the successful Actions job has a terminal non-success check-run: "
            f"conclusion={initial_conclusion!r}"
        )
    if initial_status not in PENDING_CHECK_STATUSES:
        raise ReconciliationError(f"unexpected check-run status: {initial_status!r}")

    client.complete_check_run(
        repository,
        check_run_id,
        output={
            "title": "Reconciled successful reproducibility check",
            "summary": (
                "The exact GitHub Actions job completed successfully, but its check-run "
                "was still pending; this run reconciled the hosted state."
            ),
        },
    )
    verified = client.get_check_run(repository, check_run_id)
    _check_identity(
        verified,
        job_name=job_name,
        head_sha=head_sha,
        check_run_id=check_run_id,
    )
    if verified.get("status") != "completed" or verified.get("conclusion") != "success":
        raise ReconciliationError(
            "check-run update did not produce completed/success: "
            f"status={verified.get('status')!r} conclusion={verified.get('conclusion')!r}"
        )
    return {
        **base,
        "status": "reconciled",
        "action": "patched_and_verified",
        "check_run": {
            "status": verified.get("status"),
            "conclusion": verified.get("conclusion"),
        },
    }


def _base_report(
    *,
    repository: str | None,
    run_id: int | None,
    run_attempt: int | None,
    head_sha: str | None,
    job_name: str,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "repository": repository,
        "run_id": run_id,
        "run_attempt": run_attempt,
        "head_sha": head_sha,
        "job_name": job_name,
        "fail_closed": True,
    }


def _write_report(path: Path | None, report: dict[str, Any]) -> None:
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the workflow entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY"))
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--run-attempt", type=int, default=None)
    parser.add_argument(
        "--head-sha",
        default=os.environ.get("REPRODUCIBILITY_HEAD_SHA") or os.environ.get("GITHUB_SHA"),
    )
    parser.add_argument(
        "--job-result", default=os.environ.get("REPRODUCIBILITY_JOB_RESULT", "success")
    )
    parser.add_argument("--job-name", default=DEFAULT_JOB_NAME)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fail-closed reconciliation and emit its JSON report."""

    args = build_parser().parse_args(argv)
    run_id = args.run_id if args.run_id is not None else _env_int("GITHUB_RUN_ID")
    run_attempt = (
        args.run_attempt if args.run_attempt is not None else (_env_int("GITHUB_RUN_ATTEMPT") or 1)
    )
    report = _base_report(
        repository=args.repository,
        run_id=run_id,
        run_attempt=run_attempt,
        head_sha=args.head_sha,
        job_name=args.job_name,
    )

    if args.job_result != "success":
        _write_report(
            args.output,
            {
                **report,
                "status": "not_applicable",
                "action": "none",
                "reason": f"upstream job result was {args.job_result!r}",
            },
        )
        return 0

    try:
        if not args.repository or run_id is None or not args.head_sha:
            raise ReconciliationError(
                "repository, run id, and reproducibility head SHA are required for a successful job"
            )
        result = reconcile_successful_job(
            GitHubActionsClient(),
            repository=args.repository,
            run_id=run_id,
            run_attempt=run_attempt,
            head_sha=args.head_sha,
            job_name=args.job_name,
        )
    except ReconciliationError as exc:
        _write_report(
            args.output, {**report, "status": "blocked", "action": "none", "error": str(exc)}
        )
        return 1

    _write_report(args.output, result)
    return 0


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ReconciliationError(f"{name} is not an integer: {value!r}") from exc


if __name__ == "__main__":
    sys.exit(main())
