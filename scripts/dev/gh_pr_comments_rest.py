#!/usr/bin/env python3
"""List a pull request's conversation comments through GitHub's REST API.

Why this exists
---------------
``gh pr view <number> --comments`` can fail on GitHub CLI versions that query the
retired Projects Classic GraphQL field, printing an error like::

    GraphQL: Projects (classic) is being deprecated ...
    (repository.pullRequest.projectCards)

On affected hosts the command exits ``0`` while emitting that error and returning
no comment content, so exit-code-based automation cannot detect the failure
(issue #6496, observed while reviewing #6454). This helper reads PR conversation
comments through ``GET /repos/{owner}/{repo}/issues/{number}/comments`` only.
GitHub treats pull requests as issues for this endpoint, so it returns the same
conversation-level comments ``gh pr view --comments`` would show, without ever
requesting a Projects Classic field. The PR header (title/state/url) is read from
``GET /repos/{owner}/{repo}/issues/{number}``, also pure REST.

Inline review comments (``pulls/{number}/comments``) are intentionally out of
scope; this is a drop-in for the conversation view that ``gh pr view --comments``
produces.

The helper is deliberately REST-only: authentication, authorization, malformed
responses, page-budget exhaustion, timeouts, and a missing ``gh`` CLI all fail
closed (nonzero exit, clear stderr).

Usage
-----
::

    uv run python scripts/dev/gh_pr_comments_rest.py 5220 \\
        --repo ll7/robot_sf_ll7

    # gh-like human-readable thread instead of JSON:
    uv run python scripts/dev/gh_pr_comments_rest.py 5220 --plain
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any

DEFAULT_REPO = "ll7/robot_sf_ll7"
DEFAULT_MAX_COMMENT_PAGES = 10
COMMENTS_PAGE_SIZE = 100

# Normalized comment shape, kept stable for machine consumers. Mirrors the
# comment fields exposed by scripts/dev/gh_issue_rest.py so PR and issue threads
# render the same way.
COMMENT_FIELDS = ("id", "user", "author_association", "created_at", "updated_at", "url", "body")


def _gh_api(path: str, *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    """GET *path* through ``gh api``, returning failures for clear handling."""
    args = ["gh", "api", path]
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError:
        return subprocess.CompletedProcess(
            args=args,
            returncode=127,
            stdout="",
            stderr="gh CLI not found on PATH; install GitHub CLI (https://cli.github.com/)",
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            args=args,
            returncode=124,
            stdout="",
            stderr=f"gh api timed out after {timeout} seconds; PR comments were not read",
        )


def _parse_json(result: subprocess.CompletedProcess[str], *, what: str) -> tuple[Any, str]:
    """Parse JSON from a ``gh api`` result, returning ``(data, error)``."""
    if result.returncode != 0:
        detail = result.stderr.strip() or f"gh api exited with code {result.returncode}"
        return None, f"{what} failed: {detail}"
    try:
        return json.loads(result.stdout), ""
    except json.JSONDecodeError as exc:
        snippet = result.stdout.strip()[:200]
        return None, f"{what} returned invalid JSON: {exc}; stdout snippet: {snippet!r}"


def _as_str(raw: Any) -> str:
    """Coerce a JSON value to ``str``, mapping explicit ``None`` to ``""``."""
    return "" if raw is None else str(raw)


def _normalize_comment(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize a raw REST comment payload to the stable output shape."""
    if not isinstance(raw, dict):
        raise ValueError("comment payload entry was not an object")
    raw_id = raw.get("id")
    if isinstance(raw_id, bool) or raw_id is None:
        raise ValueError("comment payload entry has no numeric id")
    try:
        comment_id = int(raw_id)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"comment payload entry has invalid id {raw_id!r}") from exc
    if comment_id < 1:
        raise ValueError(f"comment payload entry has invalid id {comment_id}")
    if not isinstance(raw.get("body"), str):
        raise ValueError("comment payload entry has no string body")
    user = raw.get("user") or {}
    if not isinstance(user, dict):
        raise ValueError("comment payload entry has an invalid user object")
    return {
        "id": comment_id,
        "user": _as_str(user.get("login")),
        "author_association": _as_str(raw.get("author_association")),
        "created_at": _as_str(raw.get("created_at")),
        "updated_at": _as_str(raw.get("updated_at")),
        "url": _as_str(raw.get("html_url", raw.get("url", ""))),
        "body": _as_str(raw.get("body")),
    }


def _normalize_pr_header(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize the issue-shaped PR header to a compact title/state/url view."""
    if not isinstance(raw.get("pull_request"), dict):
        raise ValueError("REST payload was not a pull request")
    if not isinstance(raw.get("title"), str) or not raw["title"]:
        raise ValueError("PR header payload has no title")
    if not isinstance(raw.get("state"), str) or not raw["state"]:
        raise ValueError("PR header payload has no state")
    if not isinstance(raw.get("html_url"), str) or not raw["html_url"]:
        raise ValueError("PR header payload has no html_url")
    return {
        "title": raw["title"],
        "state": raw["state"].upper(),
        "url": raw["html_url"],
    }


def fetch_pr_header(number: int, *, repo: str = DEFAULT_REPO) -> dict[str, Any]:
    """Fetch the PR header (title/state/url) via the REST issues endpoint.

    PRs are readable through ``repos/{repo}/issues/{number}`` because GitHub
    treats a PR number as an issue number. Returns ``{"status": "ok", ...}`` on
    success or ``{"status": "error", "error": ...}`` on failure.
    """
    if number < 1:
        return {"status": "error", "error": f"PR number must be positive, got {number}"}
    result = _gh_api(f"repos/{repo}/issues/{number}")
    data, error = _parse_json(result, what=f"PR {number} header")
    if error:
        return {"status": "error", "error": error}
    if not isinstance(data, dict):
        return {
            "status": "error",
            "error": f"PR {number} header payload was not an object",
        }
    try:
        payload = _normalize_pr_header(data)
    except ValueError as exc:
        return {"status": "error", "error": f"PR {number} header payload is malformed: {exc}"}
    payload["status"] = "ok"
    return payload


def fetch_pr_comments(
    number: int,
    *,
    repo: str = DEFAULT_REPO,
    max_pages: int = DEFAULT_MAX_COMMENT_PAGES,
) -> dict[str, Any]:
    """Fetch all conversation comments for PR *number* via REST.

    Paginates up to ``max_pages`` pages of ``COMMENTS_PAGE_SIZE`` comments each.
    Returns ``{"status": "ok", "comments": [...]}`` on success. Fails closed with
    ``{"status": "error", "error": ...}`` when the REST read fails, the payload
    is malformed or not a list, or the comment count exceeds the page budget (so
    workflows never silently truncate a long thread).
    """
    if number < 1:
        return {"status": "error", "error": f"PR number must be positive, got {number}"}
    if max_pages < 1:
        return {"status": "error", "error": f"max_pages must be >= 1, got {max_pages}"}
    comments: list[dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        result = _gh_api(
            f"repos/{repo}/issues/{number}/comments?per_page={COMMENTS_PAGE_SIZE}&page={page}",
        )
        data, error = _parse_json(result, what=f"comments page {page} for PR {number}")
        if error:
            return {"status": "error", "error": error}
        if not isinstance(data, list):
            return {
                "status": "error",
                "error": f"comments payload for PR {number} page {page} was not a list",
            }
        try:
            page_items = [_normalize_comment(item) for item in data]
        except ValueError as exc:
            return {
                "status": "error",
                "error": f"comments payload for PR {number} page {page} is malformed: {exc}",
            }
        comments.extend(page_items)
        if len(page_items) < COMMENTS_PAGE_SIZE:
            return {"status": "ok", "comments": comments}
    return {
        "status": "error",
        "error": (
            f"PR {number} has more than {max_pages * COMMENTS_PAGE_SIZE} comments; "
            f"increase --max-comment-pages to read the full thread"
        ),
    }


def fetch_pr_with_comments(
    number: int,
    *,
    repo: str = DEFAULT_REPO,
    max_comment_pages: int = DEFAULT_MAX_COMMENT_PAGES,
) -> dict[str, Any]:
    """Fetch the PR header together with all conversation comments via REST.

    On any failure returns ``{"number": number, "status": "error", "error": ...}``.
    """
    header = fetch_pr_header(number, repo=repo)
    if header.get("status") != "ok":
        return {"number": number, "status": "error", "error": str(header.get("error", ""))}
    comment_result = fetch_pr_comments(number, repo=repo, max_pages=max_comment_pages)
    if comment_result.get("status") != "ok":
        return {
            "number": number,
            "status": "error",
            "error": str(comment_result.get("error", "unknown comments error")),
        }
    return {
        "status": "ok",
        "number": number,
        "repo": repo,
        "title": header["title"],
        "state": header["state"],
        "url": header["url"],
        "comments": comment_result["comments"],
    }


def render_pr_comments_plain(payload: dict[str, Any]) -> str:
    """Render a normalized PR-with-comments payload as a gh-like conversation thread."""
    title = payload.get("title", "")
    state = payload.get("state", "")
    url = payload.get("url", "")
    lines: list[str] = []
    lines.append(f"title:\t{title}")
    lines.append(f"state:\t{state}")
    lines.append(f"url:\t{url}")
    lines.append("--")
    for comment in payload.get("comments", []) or []:
        author = comment.get("user", "")
        association = comment.get("author_association", "")
        created = comment.get("created_at", "")
        body = (comment.get("body", "") or "").rstrip()
        header = author
        if association:
            header = f"{author} ({association})"
        if created:
            header = f"{header} commented on {created}"
        lines.append(header)
        lines.append("--")
        lines.append(body)
        lines.append("--")
    return "\n".join(lines).rstrip() + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("number", type=int, help="Pull-request number to read.")
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help=f"owner/repo to read (default: {DEFAULT_REPO}).",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Render a gh-like human-readable conversation thread instead of JSON.",
    )
    parser.add_argument(
        "--max-comment-pages",
        type=int,
        default=DEFAULT_MAX_COMMENT_PAGES,
        help=f"Maximum comment pages to read (each {COMMENTS_PAGE_SIZE} comments).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the PR-comment reader and emit one compact JSON result or plain thread."""
    args = _build_parser().parse_args(argv)
    result = fetch_pr_with_comments(
        args.number,
        repo=args.repo,
        max_comment_pages=args.max_comment_pages,
    )
    if result.get("status") != "ok":
        print(json.dumps(result, sort_keys=True), file=sys.stderr)
        return 1
    if args.plain:
        sys.stdout.write(render_pr_comments_plain(result))
        return 0
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
