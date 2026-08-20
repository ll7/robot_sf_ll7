#!/usr/bin/env python3
"""Record an explicit post-merge audit disposition for a merged pull request.

Why this exists
---------------
Issue #7610 required that merged PRs retain an explicit post-merge audit
disposition instead of being represented as still draft/pending by stale
carriers. This helper reads the live GitHub PR object after the merge and
emits a machine-readable audit payload binding the exact merge state; with
``--comment`` it also posts a compact audit comment on the PR so the
disposition is visible next to the stale review carriers it supersedes.

Fail-closed rules: the live PR must be merged with a merge commit SHA and a
merge timestamp. Anything else (still open, unreadable, malformed) returns an
``error`` so a caller cannot record a bogus audit disposition.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from scripts.dev._gh_rest import gh_api_get as _gh_api_get
from scripts.dev._gh_rest import gh_api_post as _gh_api_post
from scripts.dev._gh_rest import parse_json as _parse_json

DEFAULT_REPO = "ll7/robot_sf_ll7"
AUDIT_SCHEMA = "post-merge-audit.v1"


def _audit_payload(number: int, *, repo: str) -> tuple[dict[str, Any] | None, str | None]:
    """Read the live PR object and build the audit payload, or an error string."""
    result = _gh_api_get(f"repos/{repo}/pulls/{number}")
    payload, error = _parse_json(result, what=f"PR {number} post-merge audit read")
    if error:
        return None, error
    if not isinstance(payload, dict):
        return None, "post-merge audit payload was not an object"

    state = payload.get("state")
    merged_at = payload.get("merged_at")
    head = payload.get("head")
    base = payload.get("base")
    merge_commit_sha = payload.get("merge_commit_sha")
    if state != "merged" or not isinstance(merged_at, str) or not merged_at:
        return None, (
            f"PR {number} is not merged (state={state!r}, merged_at={merged_at!r}); "
            "no audit disposition recorded"
        )
    if not isinstance(head, dict) or not isinstance(head.get("sha"), str):
        return None, "post-merge audit payload has no head SHA"
    if not isinstance(base, dict) or not isinstance(base.get("sha"), str):
        return None, "post-merge audit payload has no base SHA"
    if not isinstance(merge_commit_sha, str) or not merge_commit_sha:
        return None, "post-merge audit payload has no merge commit SHA"

    return (
        {
            "schema": AUDIT_SCHEMA,
            "number": number,
            "repo": repo,
            "state": "merged",
            "head_sha": head["sha"],
            "base_sha": base["sha"],
            "merge_commit_sha": merge_commit_sha,
            "merged_at": merged_at,
        },
        None,
    )


def audit_merged_pr(
    number: int, *, repo: str = DEFAULT_REPO, comment: bool = False
) -> dict[str, Any]:
    """Record and optionally comment the post-merge audit disposition for *number*."""
    if number < 1:
        return {"status": "error", "error": f"PR number must be positive, got {number}"}
    payload, error = _audit_payload(number, repo=repo)
    if error:
        return {"status": "error", "error": error}
    assert payload is not None

    result: dict[str, Any] = {"status": "ok", "audit": payload}
    if not comment:
        return result

    body = (
        "## Post-merge audit\n\n"
        f"Live PR state at audit time: `{payload['head_sha']}` -> "
        f"`{payload['base_sha']}` via merge commit `{payload['merge_commit_sha']}` "
        f"(merged {payload['merged_at']}).\n\n"
        f"Disposition: **merged, post-merge audit recorded** "
        f"(`{AUDIT_SCHEMA}`); stale review carriers on this PR are superseded."
    )
    post = _gh_api_post(f"repos/{repo}/issues/{number}/comments", {"body": body})
    if post.returncode != 0:
        detail = post.stderr.strip() or f"gh api exited with code {post.returncode}"
        return {"status": "error", "error": f"audit comment post failed: {detail}"}
    try:
        response = json.loads(post.stdout)
    except json.JSONDecodeError as exc:
        snippet = post.stdout.strip()[:200]
        return {
            "status": "error",
            "error": f"audit comment response was not valid JSON: {exc}; snippet: {snippet!r}",
        }
    if not isinstance(response, dict):
        return {"status": "error", "error": "audit comment response was not an object"}
    result["comment_url"] = str(response.get("html_url", ""))
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("number", type=int, help="Merged PR number to audit.")
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help=f"owner/repo to audit (default: {DEFAULT_REPO}).",
    )
    parser.add_argument(
        "--comment",
        action="store_true",
        help="Post a compact post-merge audit comment on the PR.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the audit helper and emit one compact JSON result."""
    args = _build_parser().parse_args(argv)
    result = audit_merged_pr(args.number, repo=args.repo, comment=args.comment)
    stream = sys.stdout if result["status"] == "ok" else sys.stderr
    print(json.dumps(result, sort_keys=True), file=stream)
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
