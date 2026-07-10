#!/usr/bin/env python3
"""Print GitHub Actions job logs, falling back to check-run annotations.

Some infrastructure failures make a completed GitHub Actions job unavailable
through both ``gh run view --log`` and the job-log REST endpoint. GitHub still
attaches the actionable error to the job's check-run annotations. This helper
uses the job metadata to find that check run and prints those annotations when
the normal log command returns no usable output.

Example::

    uv run python scripts/dev/diagnose_actions_job.py 86418927103
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any

DEFAULT_REPO = "ll7/robot_sf_ll7"
API_PREFIX = "https://api.github.com/"


def _gh(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run ``gh`` without raising so diagnostic fallback remains available."""
    try:
        return subprocess.run(
            ["gh", *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return subprocess.CompletedProcess(
            args=["gh", *args],
            returncode=127,
            stdout="",
            stderr="gh CLI not found on PATH; install GitHub CLI (https://cli.github.com/)",
        )


def _parse_json(result: subprocess.CompletedProcess[str], *, source: str) -> dict[str, Any] | None:
    """Return a JSON object or print a concise failure for the failed source."""
    if result.returncode != 0:
        detail = result.stderr.strip() or f"gh exited with code {result.returncode}"
        print(f"Could not read {source}: {detail}", file=sys.stderr)
        return None
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        print(f"Could not parse {source} JSON: {exc}", file=sys.stderr)
        return None
    if not isinstance(payload, dict):
        print(f"Could not read {source}: expected a JSON object", file=sys.stderr)
        return None
    return payload


def _parse_annotation_pages(
    result: subprocess.CompletedProcess[str],
) -> list[dict[str, Any]] | None:
    """Flatten ``gh api --paginate --slurp`` annotations or reject an empty response."""
    if result.returncode != 0:
        detail = result.stderr.strip() or f"gh exited with code {result.returncode}"
        print(f"Could not recover check-run annotations: {detail}", file=sys.stderr)
        return None
    try:
        pages = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        print(f"Could not parse check-run annotations JSON: {exc}", file=sys.stderr)
        return None
    if not isinstance(pages, list) or not all(isinstance(page, list) for page in pages):
        print(
            "Could not recover check-run annotations: expected paginated JSON lists",
            file=sys.stderr,
        )
        return None
    annotations = [item for page in pages for item in page]
    if not annotations:
        print(
            "Could not recover check-run annotations: the endpoint returned no annotations",
            file=sys.stderr,
        )
        return None
    return annotations


def _annotations_path(check_run_url: object) -> str | None:
    """Convert GitHub's absolute check-run API URL into a ``gh api`` path."""
    if not isinstance(check_run_url, str) or not check_run_url.startswith(API_PREFIX):
        return None
    path = check_run_url.removeprefix(API_PREFIX).rstrip("/")
    if "/check-runs/" not in path:
        return None
    return f"{path}/annotations?per_page=100"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the Actions job identifier and optional repository override."""
    parser = argparse.ArgumentParser(
        description="Print an Actions job log or its check-run annotations when logs are absent.",
    )
    parser.add_argument("job_id", type=int, help="GitHub Actions workflow job ID.")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub repository as OWNER/REPO.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Print normal logs first, then fail closed after an annotation fallback failure."""
    args = _parse_args(argv)
    job_result = _gh(["api", f"repos/{args.repo}/actions/jobs/{args.job_id}"])
    job = _parse_json(job_result, source=f"job metadata for {args.job_id}")
    if job is None:
        return 1

    run_id = job.get("run_id")
    if not isinstance(run_id, int):
        print(
            f"Job {args.job_id} metadata has no integer run_id; cannot read normal logs.",
            file=sys.stderr,
        )
        return 1

    log_result = _gh(
        ["run", "view", str(run_id), "--repo", args.repo, "--job", str(args.job_id), "--log"],
    )
    if log_result.returncode == 0 and log_result.stdout.strip():
        sys.stdout.write(log_result.stdout)
        return 0

    detail = log_result.stderr.strip() or "the command returned no log output"
    print(f"Normal log retrieval unavailable for job {args.job_id}: {detail}", file=sys.stderr)
    annotations_path = _annotations_path(job.get("check_run_url"))
    if annotations_path is None:
        print(
            f"Job {args.job_id} metadata has no usable check_run_url; diagnostics unavailable.",
            file=sys.stderr,
        )
        return 1

    print("Falling back to check-run annotations.", file=sys.stderr)
    annotations_result = _gh(["api", "--paginate", "--slurp", annotations_path])
    annotations = _parse_annotation_pages(annotations_result)
    if annotations is None:
        return 1
    print(json.dumps(annotations))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
