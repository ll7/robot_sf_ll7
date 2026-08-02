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
import re
import subprocess
import sys
from typing import Any

DEFAULT_REPO = "ll7/robot_sf_ll7"
API_PREFIX = "https://api.github.com/"
# Safety guard for the rel="next" pagination loop. The check-run annotations
# endpoint permits pages of up to 100 results; this is a local request budget.
MAX_ANNOTATION_PAGES = 100


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


def _split_include_output(stdout: str) -> tuple[str, str]:
    """Split ``gh api --include`` stdout into ``(headers block, body)``.

    ``gh api --include`` prints the HTTP status line and headers, a blank line,
    then the JSON body. Header lines use CRLF; the separator is also accepted as
    a bare LF so the parser stays robust across GitHub CLI versions.
    """
    for sep in ("\r\n\r\n", "\n\n"):
        index = stdout.find(sep)
        if index != -1:
            return stdout[:index], stdout[index + len(sep) :]
    return "", stdout


def _header_value(headers_block: str, name: str) -> str | None:
    """Return the joined values of a named header from a ``--include`` block."""
    wanted = name.lower()
    values: list[str] = []
    for line in headers_block.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        if key.strip().lower() == wanted:
            values.append(value.strip())
    return ", ".join(values) if values else None


def _next_link(headers_block: str) -> str | None:
    """Return the ``rel="next"`` URL from a ``gh api --include`` headers block."""
    link_value = _header_value(headers_block, "Link")
    if link_value is None:
        return None
    for entry in link_value.split(","):
        match = re.search(r"<([^>]+)>", entry)
        if match and 'rel="next"' in entry:
            return match.group(1)
    return None


def _fetch_annotation_page(
    request_path: str,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Fetch one annotation page via ``gh api --include``.

    Returns ``(annotations, next_url)``: ``annotations`` is the page's JSON array
    (possibly empty), and ``next_url`` is the ``rel="next"`` Link URL or
    ``None``. On any ``gh`` failure or unparseable response, ``annotations`` is
    ``None`` after a concise error is printed to stderr.
    """
    result = _gh(["api", "--include", request_path])
    if result.returncode != 0:
        detail = result.stderr.strip() or f"gh exited with code {result.returncode}"
        print(f"Could not recover check-run annotations: {detail}", file=sys.stderr)
        return None, None
    headers_block, body = _split_include_output(result.stdout)
    if not headers_block:
        print(
            "Could not recover check-run annotations: expected HTTP headers from gh api --include",
            file=sys.stderr,
        )
        return None, None
    try:
        page = json.loads(body)
    except json.JSONDecodeError as exc:
        print(f"Could not parse check-run annotations JSON: {exc}", file=sys.stderr)
        return None, None
    if not isinstance(page, list):
        print(
            "Could not recover check-run annotations: expected a JSON array per page",
            file=sys.stderr,
        )
        return None, None
    annotations: list[dict[str, Any]] = []
    for annotation in page:
        if not isinstance(annotation, dict):
            print(
                "Could not recover check-run annotations: expected annotation objects",
                file=sys.stderr,
            )
            return None, None
        annotations.append(annotation)
    return annotations, _next_link(headers_block)


def _collect_annotations(initial_path: str) -> list[dict[str, Any]] | None:
    """Concatenate check-run annotations across ``rel="next"`` pages.

    Follows the REST pagination chain exposed by ``gh api --include`` so the
    helper works across GitHub CLI versions (including those that reject the
    multi-page aggregation flag): each request returns one JSON array of
    annotations, and the ``rel="next"`` Link URL is requested until the chain
    ends. Returns ``None`` (after a stderr error) when any page fails, the
    response is malformed, or no annotations are returned.
    """
    annotations: list[dict[str, Any]] = []
    request_path: str | None = initial_path
    for _ in range(MAX_ANNOTATION_PAGES):
        if request_path is None:
            break
        page, next_url = _fetch_annotation_page(request_path)
        if page is None:
            return None
        annotations.extend(page)
        request_path = next_url
        if request_path is None:
            break
    else:
        print(
            "Could not recover check-run annotations: pagination exceeded the page guard",
            file=sys.stderr,
        )
        return None
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
    annotations = _collect_annotations(annotations_path)
    if annotations is None:
        return 1
    print(json.dumps(annotations))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
