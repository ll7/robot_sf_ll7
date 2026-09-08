#!/usr/bin/env python3
"""Diagnose exact GitHub Actions jobs without retrying or changing CI state.

Some infrastructure failures make a completed GitHub Actions job unavailable
through the job-log REST endpoint. GitHub still
attaches the actionable error to the job's check-run annotations. This helper
uses the job metadata to find that check run and prints those annotations when
the exact log endpoint returns no usable output. Opt-in JSON classifies a narrow
artifact-finalization failure; successful retrieval does not mean successful CI.

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
from urllib.parse import urlsplit

DEFAULT_REPO = "ll7/robot_sf_ll7"
API_PREFIX = "https://api.github.com/"
DEFAULT_GH_TIMEOUT_SECONDS = 30
# Safety guard for the rel="next" pagination loop. The check-run annotations
# endpoint permits pages of up to 100 results; this is a local request budget.
MAX_ANNOTATION_PAGES = 100
MAX_EXCERPT_CHARS = 2000
# Deliberately match the observed error record, not independent phase/status words.
FINALIZATION_403 = (
    "Failed to FinalizeArtifact: Received non-retryable error: Failed request: (403) Forbidden: "
    'Error from intermediary with HTTP status code 403 "Forbidden"'
)


def _gh(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run ``gh`` without raising so diagnostic fallback remains available."""
    command = ["gh", *args]
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=DEFAULT_GH_TIMEOUT_SECONDS,
        )
    except FileNotFoundError:
        return subprocess.CompletedProcess(
            args=command,
            returncode=127,
            stdout="",
            stderr="gh CLI not found on PATH; install GitHub CLI (https://cli.github.com/)",
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            args=command,
            returncode=124,
            stdout="",
            stderr=f"gh command timed out after {DEFAULT_GH_TIMEOUT_SECONDS} seconds",
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
    next_url = _next_link(headers_block)
    if 'rel="next"' in (_header_value(headers_block, "Link") or "") and next_url is None:
        print("Could not recover check-run annotations: malformed next-page link", file=sys.stderr)
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
    return annotations, next_url


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
    expected_path = "/" + initial_path.removeprefix(API_PREFIX).split("?", 1)[0]
    for _ in range(MAX_ANNOTATION_PAGES):
        if request_path is None:
            break
        page, next_url = _fetch_annotation_page(request_path)
        if page is None:
            return None
        if next_url is not None:
            try:
                target = urlsplit(next_url)
            except ValueError:
                print(
                    "Could not recover check-run annotations: malformed next-page URL",
                    file=sys.stderr,
                )
                return None
            if (target.scheme, target.netloc, target.path, target.fragment) != (
                "https",
                "api.github.com",
                expected_path,
                "",
            ):
                print(
                    "Could not recover check-run annotations: pagination changed check-run identity",
                    file=sys.stderr,
                )
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
    parser.add_argument("--json", action="store_true", help="Emit actions_job_diagnostic.v1 JSON.")
    return parser.parse_args(argv)


def _valid_job(job: dict[str, Any], repo: str, job_id: int) -> bool:
    """Require exact identity from the requested job, never the latest run attempt."""
    return (
        all(type(job.get(key)) is int and job[key] > 0 for key in ("id", "run_id", "run_attempt"))
        and job["id"] == job_id
        and isinstance(job.get("url"), str)
        and job["url"].casefold() == f"{API_PREFIX}repos/{repo}/actions/jobs/{job_id}".casefold()
        and isinstance(job.get("head_sha"), str)
        and re.fullmatch(r"[0-9a-fA-F]{40}", job["head_sha"]) is not None
        and isinstance(job.get("status"), str)
        and bool(job["status"].strip())
        and "conclusion" in job
        and (job["conclusion"] is None or isinstance(job["conclusion"], str))
    )


def _emit_json(
    args: argparse.Namespace,
    job: dict[str, Any] | None,
    *,
    source: str | None = None,
    endpoint: str | None = None,
    records: list[str] | None = None,
    reason: str | None = None,
) -> None:
    """Print one bounded envelope, keeping diagnostic and original job outcomes separate."""
    job = job or {}
    record_number = None
    excerpt = ""
    classification = None
    for number, record in enumerate(records or [], start=1):
        # Each log line / annotation message is independent. Even within one
        # annotation, do not join lines to manufacture the observed signature.
        if any(FINALIZATION_403 in line for line in record.splitlines()):
            classification = "artifact_finalization_403"
            record_number, excerpt = number, record
            break
    if records and record_number is None:
        record_number, excerpt = 1, records[0]
    offset = max(0, excerpt.find(FINALIZATION_403) - 120) if classification else 0
    payload = {
        "schema": "actions_job_diagnostic.v1",
        "repository": args.repo,
        "job_id": args.job_id,
        "run_id": job.get("run_id"),
        "run_attempt": job.get("run_attempt"),
        "head_sha": job.get("head_sha"),
        "job_status": job.get("status"),
        "job_conclusion": job.get("conclusion"),
        "diagnostic_status": "unavailable"
        if reason
        else "matched"
        if classification
        else "unmatched",
        "classification": classification,
        "reason": reason,
        "artifact_publication": "unconfirmed",
        "evidence": {
            "source": source,
            "endpoint": endpoint,
            "record": record_number,
            "excerpt": excerpt[offset : offset + MAX_EXCERPT_CHARS],
            "truncated": offset > 0 or len(excerpt) > MAX_EXCERPT_CHARS,
        },
    }
    print(json.dumps(payload))


def main(argv: list[str] | None = None) -> int:
    """Print normal logs first, then fail closed after an annotation fallback failure."""
    args = _parse_args(argv)
    if args.job_id <= 0 or re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repo) is None:
        print("Expected a positive job ID and repository OWNER/REPO.", file=sys.stderr)
        if args.json:
            _emit_json(args, None, reason="invalid_request")
        return 1
    job_result = _gh(["api", f"repos/{args.repo}/actions/jobs/{args.job_id}"])
    job = _parse_json(job_result, source=f"job metadata for {args.job_id}")
    if job is None:
        if args.json:
            _emit_json(args, None, reason="job_metadata_unavailable")
        return 1

    if not _valid_job(job, args.repo, args.job_id):
        print(
            f"Job {args.job_id} metadata has missing or mismatched identity/status; diagnostics unavailable.",
            file=sys.stderr,
        )
        if args.json:
            _emit_json(args, None, reason="job_metadata_identity_invalid")
        return 1

    log_path = f"repos/{args.repo}/actions/jobs/{args.job_id}/logs"
    log_result = _gh(["api", log_path])
    if log_result.returncode == 0 and log_result.stdout.strip():
        if args.json:
            _emit_json(
                args,
                job,
                source="job_logs",
                endpoint=log_path,
                records=log_result.stdout.splitlines(),
            )
        else:
            sys.stdout.write(log_result.stdout)
        return 0

    detail = log_result.stderr.strip() or "the command returned no log output"
    print(f"Normal log retrieval unavailable for job {args.job_id}: {detail}", file=sys.stderr)
    return _annotation_fallback(args, job)


def _annotation_fallback(args: argparse.Namespace, job: dict[str, Any]) -> int:
    """Recover complete, same-check-run annotations or report unavailable diagnostics."""
    annotations_path = _annotations_path(job.get("check_run_url"))
    if annotations_path is None or not re.fullmatch(
        rf"repos/{re.escape(args.repo)}/check-runs/[1-9][0-9]*/annotations\?per_page=100",
        annotations_path,
        flags=re.IGNORECASE,
    ):
        print(
            f"Job {args.job_id} metadata has no usable check_run_url; diagnostics unavailable.",
            file=sys.stderr,
        )
        if args.json:
            _emit_json(args, job, reason="check_run_identity_invalid")
        return 1

    print("Falling back to check-run annotations.", file=sys.stderr)
    annotations = _collect_annotations(annotations_path)
    if annotations is None:
        if args.json:
            _emit_json(
                args,
                job,
                source="check_run_annotations",
                endpoint=annotations_path,
                reason="annotations_unavailable_or_incomplete",
            )
        return 1
    messages = [annotation.get("message") for annotation in annotations]
    if not all(isinstance(message, str) and message.strip() for message in messages):
        print("Could not recover check-run annotations: missing usable messages", file=sys.stderr)
        if args.json:
            _emit_json(
                args,
                job,
                source="check_run_annotations",
                endpoint=annotations_path,
                reason="annotation_messages_unusable",
            )
        return 1
    if args.json:
        _emit_json(
            args, job, source="check_run_annotations", endpoint=annotations_path, records=messages
        )
    else:
        print(json.dumps(annotations))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
