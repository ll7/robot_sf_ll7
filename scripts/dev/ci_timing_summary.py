"""Summarize GitHub Actions CI run timing from ``gh run view`` JSON."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

_GH_RUN_FIELDS = "databaseId,displayTitle,headBranch,status,conclusion,createdAt,updatedAt,jobs"


@dataclass(frozen=True, slots=True)
class StepTiming:
    """Duration for one GitHub Actions job step."""

    name: str
    duration_seconds: float
    started_at: str
    completed_at: str


@dataclass(frozen=True, slots=True)
class JobTiming:
    """Duration for one GitHub Actions job."""

    name: str
    duration_seconds: float
    started_at: str
    completed_at: str


@dataclass(frozen=True, slots=True)
class RunTimingSummary:
    """Compact timing summary for one GitHub Actions workflow run."""

    run_id: int
    title: str
    queue_seconds: float
    job_seconds: float
    total_seconds: float
    slowest_jobs: list[JobTiming]
    slowest_steps: list[StepTiming]

    def to_json(self) -> str:
        """Serialize the summary as deterministic JSON."""
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def _parse_timestamp(value: str | None) -> datetime | None:
    """Parse a GitHub timestamp, treating missing or zero timestamps as absent."""
    if not value or value.startswith("0001-01-01"):
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _duration_seconds(start: str | None, end: str | None) -> float | None:
    """Return elapsed seconds for a timestamp pair, or ``None`` if incomplete."""
    started = _parse_timestamp(start)
    completed = _parse_timestamp(end)
    if started is None or completed is None:
        return None
    return max((completed - started).total_seconds(), 0.0)


def _completed_step_timings(payload: dict[str, Any]) -> list[StepTiming]:
    """Extract completed step durations from all jobs in a run payload."""
    timings: list[StepTiming] = []
    for job in payload.get("jobs", []):
        for step in job.get("steps", []):
            duration = _duration_seconds(step.get("startedAt"), step.get("completedAt"))
            if duration is None:
                continue
            timings.append(
                StepTiming(
                    name=str(step.get("name", "")),
                    duration_seconds=duration,
                    started_at=str(step.get("startedAt", "")),
                    completed_at=str(step.get("completedAt", "")),
                ),
            )
    return timings


def _completed_job_timings(payload: dict[str, Any]) -> list[JobTiming]:
    """Extract completed job durations from a run payload."""
    timings: list[JobTiming] = []
    for job in payload.get("jobs", []):
        duration = _duration_seconds(job.get("startedAt"), job.get("completedAt"))
        if duration is None:
            continue
        timings.append(
            JobTiming(
                name=str(job.get("name", "")),
                duration_seconds=duration,
                started_at=str(job.get("startedAt", "")),
                completed_at=str(job.get("completedAt", "")),
            ),
        )
    return timings


def summarize_run(payload: dict[str, Any], *, top: int = 10) -> RunTimingSummary:
    """Summarize queue, job, total, slowest-job, and slowest-step durations."""
    jobs = list(payload.get("jobs", []))
    created = _parse_timestamp(payload.get("createdAt"))
    updated = _parse_timestamp(payload.get("updatedAt"))
    job_starts = [
        parsed for job in jobs if (parsed := _parse_timestamp(job.get("startedAt"))) is not None
    ]
    job_completions = [
        parsed for job in jobs if (parsed := _parse_timestamp(job.get("completedAt"))) is not None
    ]

    queue_seconds = 0.0
    if created is not None and job_starts:
        queue_seconds = max((min(job_starts) - created).total_seconds(), 0.0)

    job_seconds = 0.0
    if job_starts and job_completions:
        job_seconds = max((max(job_completions) - min(job_starts)).total_seconds(), 0.0)

    total_seconds = 0.0
    if created is not None and updated is not None:
        total_seconds = max((updated - created).total_seconds(), 0.0)

    slowest_steps = sorted(
        _completed_step_timings(payload),
        key=lambda step: (-step.duration_seconds, step.name),
    )[:top]
    slowest_jobs = sorted(
        _completed_job_timings(payload),
        key=lambda job: (-job.duration_seconds, job.name),
    )[:top]

    return RunTimingSummary(
        run_id=int(payload.get("databaseId") or 0),
        title=str(payload.get("displayTitle") or ""),
        queue_seconds=queue_seconds,
        job_seconds=job_seconds,
        total_seconds=total_seconds,
        slowest_jobs=slowest_jobs,
        slowest_steps=slowest_steps,
    )


def _format_seconds(seconds: float) -> str:
    """Format seconds with one decimal place for compact tables."""
    return f"{seconds:.1f}s"


def format_markdown(summary: RunTimingSummary) -> str:
    """Format a timing summary as Markdown for issues and PR comments."""
    lines = [
        f"# CI Timing Summary: Run {summary.run_id}",
        "",
        f"Title: {summary.title}",
        "",
        "| metric | duration |",
        "| --- | --- |",
        f"| queue | {_format_seconds(summary.queue_seconds)} |",
        f"| job | {_format_seconds(summary.job_seconds)} |",
        f"| total | {_format_seconds(summary.total_seconds)} |",
        "",
        "## Slowest jobs",
        "| job | duration | started | completed |",
        "| --- | --- | --- | --- |",
    ]
    for job in summary.slowest_jobs:
        lines.append(
            "| "
            f"{job.name} | {_format_seconds(job.duration_seconds)} | "
            f"{job.started_at} | {job.completed_at} |",
        )

    lines.extend(
        [
            "",
            "## Slowest steps",
            "| step | duration | started | completed |",
            "| --- | --- | --- | --- |",
        ]
    )
    if not summary.slowest_steps:
        lines.append("| _No step timestamps reported by `gh run view`_ | n/a | n/a | n/a |")
        return "\n".join(lines)

    for step in summary.slowest_steps:
        lines.append(
            "| "
            f"{step.name} | {_format_seconds(step.duration_seconds)} | "
            f"{step.started_at} | {step.completed_at} |",
        )
    return "\n".join(lines)


def _load_payload_from_gh(run_id: str) -> dict[str, Any]:
    """Fetch a workflow run payload through the GitHub CLI."""
    output = subprocess.check_output(
        ["gh", "run", "view", run_id, "--json", _GH_RUN_FIELDS],
        text=True,
    )
    return json.loads(output)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse command-line arguments for the timing summary tool."""
    parser = argparse.ArgumentParser(description="Summarize GitHub Actions CI timing.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-id", help="GitHub Actions run id to fetch through gh.")
    source.add_argument("--run-json", type=Path, help="Saved gh run view JSON payload.")
    parser.add_argument("--top", type=int, default=10, help="Number of slowest steps to report.")
    parser.add_argument("--json", action="store_true", help="Print deterministic JSON.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for CI timing summaries."""
    args = _parse_args(argv)
    if args.top <= 0:
        raise SystemExit("--top must be a positive integer")
    payload = (
        json.loads(args.run_json.read_text(encoding="utf-8"))
        if args.run_json is not None
        else _load_payload_from_gh(args.run_id)
    )
    summary = summarize_run(payload, top=args.top)
    sys.stdout.write(summary.to_json() if args.json else format_markdown(summary))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
