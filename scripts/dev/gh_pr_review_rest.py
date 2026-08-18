#!/usr/bin/env python3
"""Publish an exact-head pull-request review through the REST API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from scripts.dev._gh_rest import gh_api_review_post as _gh_api_post
from scripts.dev._gh_rest import parse_json as _parse_json
from scripts.dev.pr_write_guard import DEFAULT_REPO, guard_pr_write, pr_write_lock

REVIEW_EVENTS = ("COMMENT", "APPROVE", "REQUEST_CHANGES")


def _read_body_file(body_file: Path) -> tuple[str | None, str | None]:
    """Read a non-empty UTF-8 review body file."""
    try:
        body = body_file.read_text(encoding="utf-8")
    except OSError as exc:
        return None, f"could not read review body file {body_file}: {exc}"
    if not body.strip():
        return None, "review body file must not be empty"
    return body, None


def post_review(
    number: int,
    body_file: Path,
    *,
    expected_head_sha: str,
    event: str = "COMMENT",
    repo: str = DEFAULT_REPO,
) -> dict[str, Any]:
    """Post one review only when the PR is still open at the expected head."""
    body, body_error = _read_body_file(body_file)
    if body_error:
        return {"status": "error", "error": body_error}
    assert body is not None
    event = event.upper()
    if event not in REVIEW_EVENTS:
        return {
            "status": "error",
            "error": f"event must be one of {', '.join(REVIEW_EVENTS)}",
        }

    try:
        with pr_write_lock(repo, number):
            guard = guard_pr_write(
                number,
                repo=repo,
                expected_head_sha=expected_head_sha,
                operation="commented_review" if event == "COMMENT" else "review",
            )
            if guard["status"] != "ok":
                return guard
            result = _gh_api_post(
                f"repos/{repo}/pulls/{number}/reviews",
                {"body": body, "event": event, "commit_id": expected_head_sha},
            )
            payload, error = _parse_json(result, what=f"PR {number} review publication")
            if error:
                return {"status": "error", "error": error}
            if not isinstance(payload, dict):
                return {"status": "error", "error": "review response was not an object"}
            response_commit = payload.get("commit_id")
            if response_commit and str(response_commit).lower() != expected_head_sha.lower():
                return {
                    "status": "error",
                    "error": "review response was bound to a different commit",
                    "expected_head_sha": expected_head_sha,
                    "observed_review_commit_id": response_commit,
                }
            review_id = payload.get("id")
            if isinstance(review_id, bool) or not isinstance(review_id, int) or review_id < 1:
                return {"status": "error", "error": "review response had no numeric id"}
            return {
                "status": "ok",
                "number": number,
                "repo": repo,
                "event": event,
                "head_sha": expected_head_sha,
                "review_id": review_id,
                "url": str(payload.get("html_url", "")),
            }
    except RuntimeError as exc:
        return {"status": "error", "error": str(exc)}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("number", type=int, help="Pull-request number to review.")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="owner/repo to update.")
    parser.add_argument("--body-file", type=Path, required=True, help="Review body file.")
    parser.add_argument(
        "--expected-head-sha",
        required=True,
        help="Full 40-character PR head SHA captured by the review lane.",
    )
    parser.add_argument("--event", choices=REVIEW_EVENTS, default="COMMENT")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the guarded REST review writer and emit one JSON result."""
    args = _build_parser().parse_args(argv)
    result = post_review(
        args.number,
        args.body_file,
        expected_head_sha=args.expected_head_sha,
        event=args.event,
        repo=args.repo,
    )
    status = result.get("status")
    print(json.dumps(result, sort_keys=True), file=sys.stdout if status == "ok" else sys.stderr)
    if status == "ok":
        return 0
    if status == "review_skipped_stale_state":
        return 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
