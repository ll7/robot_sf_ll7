#!/usr/bin/env python3
"""Shared GitHub issue-with-comments helper with a fail-closed REST fallback.

Why this exists
---------------
``gh issue view <number> --comments`` requests the deprecated
``repository.issue.projectCards`` GraphQL field, which fails on some GitHub CLI
versions with an error like::

    GraphQL: Projects (classic) is being deprecated ... (repository.issue.projectCards)

That breaks autonomous workflows that read an issue and its comments before
editing (see issues #5021 and #5092). The ``thread`` command first tries the
concise GitHub CLI route and falls back to paginated REST reads only for the
known GraphQL-path failures: the deprecated ``repository.issue.projectCards``
field, GraphQL API rate-limit exhaustion, secondary rate limits, and other
GraphQL errors. Authentication, authorization, and repository-resolution failures
stay fail-closed. The existing ``view`` command and library functions remain
explicit REST-backed interfaces.

Public library API
------------------
- :func:`fetch_issue`           -- normalized issue body via REST
- :func:`fetch_comments`        -- all comments via REST (paginated)
- :func:`fetch_issue_with_comments` -- combined issue + comments payload
- :func:`render_issue_plain`    -- gh-like human-readable thread rendering
- :func:`read_complete_issue_thread` -- native read with targeted REST fallback

CLI
---
::

    python scripts/dev/gh_issue_rest.py thread <number> [--repo <owner/repo>]
        [--max-comment-pages N]

    python scripts/dev/gh_issue_rest.py view <number> [--repo <owner/repo>]
        [--comments] [--json <fields>] [--plain] [--max-comment-pages N]

Use ``thread`` as a drop-in replacement for ``gh issue view <number> --comments``
in autonomous workflows. It fails closed (nonzero exit, clear stderr) when a
non-matching native error occurs or when the REST fallback cannot read the full
thread.

Field normalization
-------------------
REST returns lowercase ``state`` (``open``/``closed``) and ``html_url``. To be a
drop-in for ``gh issue view --json`` consumers, the normalized output exposes
``state`` uppercased (``OPEN``/``CLOSED``) and ``url`` equal to ``html_url``.
Comment entries use ``user`` (login) and ``url`` (``html_url``) for the same
reason.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any
from urllib.parse import urlsplit

from scripts.dev import github_transport_policy as _transport_policy
from scripts.dev._gh_rest import as_str as _as_str
from scripts.dev._gh_rest import parse_json as _parse_json
from scripts.dev._gh_rest import run_gh_api as _gh_api
from scripts.dev.github_transport_policy import (
    get_transport_contract,
    is_fallback_eligible,
)

DEFAULT_REPO = "ll7/robot_sf_ll7"
DEFAULT_MAX_COMMENT_PAGES = 10
COMMENTS_PAGE_SIZE = 100
VALID_ISSUE_STATES = frozenset({"OPEN", "CLOSED"})
PROJECT_CARDS_ERROR_MARKER = _transport_policy.PROJECT_CARDS_ERROR_MARKER
FALLBACK_ELIGIBLE_MARKERS = _transport_policy.FALLBACK_ELIGIBLE_MARKERS
FAIL_CLOSED_ERROR_MARKERS = _transport_policy.FAIL_CLOSED_ERROR_MARKERS
TRANSPORT_CONTRACT = get_transport_contract("gh_issue_rest.py")

# Fields exposed in the normalized issue payload, in a stable order.
ISSUE_FIELDS = (
    "number",
    "title",
    "body",
    "state",
    "url",
    "is_pull_request",
    "user",
    "author_association",
    "labels",
    "assignees",
    "created_at",
    "updated_at",
)
COMMENT_FIELDS = ("id", "user", "author_association", "created_at", "updated_at", "url", "body")


def _repo_parts(repo: str) -> tuple[str, str]:
    """Return the owner and repository name from a canonical owner/name value."""
    if not isinstance(repo, str):
        raise ValueError("repository must be a string in OWNER/REPO form")
    parts = repo.split("/")
    if len(parts) != 2 or any(not part for part in parts):
        raise ValueError(f"repository must be in OWNER/REPO form, got {repo!r}")
    return parts[0], parts[1]


def validate_issue_identity(payload: object, *, repo: str, number: int) -> None:
    """Validate normalized issue identity before a caller can skip or write.

    The REST issues endpoint serves both issues and pull requests. Callers must
    use the explicit is_pull_request discriminator and a canonical URL that
    agrees with the requested repository, number, and resource kind; malformed
    or unknown identity is never a safe no-op.
    """
    if not isinstance(payload, dict):
        raise ValueError("issue result must be an object")
    if "status" in payload and payload.get("status") != "ok":
        raise ValueError(f"issue result status must be ok, got {payload.get('status')!r}")
    if type(number) is not int or number < 1:
        raise ValueError(f"requested issue number must be a positive integer, got {number!r}")

    raw_number = payload.get("number")
    if type(raw_number) is not int or raw_number != number:
        raise ValueError(
            f"issue identity number does not match requested issue ({raw_number!r} != {number})"
        )
    state = payload.get("state")
    if not isinstance(state, str) or state not in VALID_ISSUE_STATES:
        raise ValueError(f"issue identity state must be OPEN or CLOSED, got {state!r}")
    is_pull_request = payload.get("is_pull_request")
    if type(is_pull_request) is not bool:
        raise ValueError("issue identity is_pull_request must be a boolean")
    raw_url = payload.get("url")
    if not isinstance(raw_url, str) or not raw_url:
        raise ValueError("issue identity URL must be a non-empty string")

    owner, repository = _repo_parts(repo)
    parsed = urlsplit(raw_url)
    if parsed.scheme != "https" or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError(f"issue identity URL is not canonical: {raw_url!r}")
    resource = "pull" if is_pull_request else "issues"
    expected_path = f"/{owner}/{repository}/{resource}/{number}"
    if parsed.path.casefold() != expected_path.casefold():
        raise ValueError(
            "issue identity URL does not match requested repository, number, or resource kind: "
            f"{raw_url!r}"
        )


def _gh_issue_view(
    number: int, *, repo: str = DEFAULT_REPO, timeout: int = 30
) -> subprocess.CompletedProcess:
    """Run the concise native complete-thread read without raising on missing ``gh``."""
    args = ["gh", "issue", "view", str(number), "--repo", repo, "--comments"]
    try:
        # ``gh issue view`` renders nothing when stdout is not a terminal.  Force
        # its normal human-readable output so a successful native read is never
        # mistaken for an empty complete thread in automation.
        env = {**os.environ, "GH_FORCE_TTY": "100%", "GH_PAGER": "cat", "NO_COLOR": "1"}
        return subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=env,
        )
    except FileNotFoundError:
        return subprocess.CompletedProcess(
            args=args,
            returncode=127,
            stdout="",
            stderr="gh CLI not found on PATH; install GitHub CLI (https://cli.github.com/)",
        )


def _normalize_state(raw: Any) -> str:
    """Return an uppercase state string matching ``gh issue view --json``."""
    return str(raw).upper() if raw else ""


def _validate_named_objects(value: Any, *, field: str, key: str) -> None:
    """Validate a REST list whose entries must expose one non-empty string key."""
    if not isinstance(value, list):
        raise ValueError(f"issue payload {field} must be a list")
    if any(
        not isinstance(item, dict) or not isinstance(item.get(key), str) or not item[key]
        for item in value
    ):
        raise ValueError(f"issue payload {field} must contain named objects")


def _validate_issue_payload(raw: dict[str, Any]) -> None:
    """Validate the required fields used by the normalized issue contract."""
    raw_number = raw.get("number")
    if type(raw_number) is not int or raw_number < 1:
        raise ValueError("issue payload number must be a positive integer")
    title = raw.get("title")
    if not isinstance(title, str) or not title:
        raise ValueError("issue payload title must be a non-empty string")
    if "body" not in raw:
        raise ValueError("issue payload body field is missing")
    body = raw["body"]
    if body is not None and not isinstance(body, str):
        raise ValueError("issue payload body must be a string or null")
    state = raw.get("state")
    if not isinstance(state, str) or not state.strip():
        raise ValueError("issue payload state must be a non-empty string")
    raw_url = raw.get("html_url")
    if not isinstance(raw_url, str) or not raw_url:
        raise ValueError("issue payload html_url must be a non-empty string")
    _validate_named_objects(raw.get("labels"), field="labels", key="name")
    _validate_named_objects(raw.get("assignees"), field="assignees", key="login")
    raw_user = raw.get("user")
    if raw_user is not None and not isinstance(raw_user, dict):
        raise ValueError("issue payload user must be an object or null")
    if "pull_request" in raw and not isinstance(raw["pull_request"], dict):
        raise ValueError("issue payload pull_request must be an object when present")


def _normalize_issue(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize a raw REST issue payload to the stable output shape."""
    _validate_issue_payload(raw)
    raw_number = raw["number"]
    title = raw["title"]
    body = raw["body"]
    state = raw["state"]
    raw_url = raw["html_url"]
    raw_labels = raw["labels"]
    raw_assignees = raw["assignees"]
    raw_user = raw.get("user")
    is_pull_request = "pull_request" in raw
    labels = sorted(label["name"] for label in raw_labels)
    assignees = sorted(user["login"] for user in raw_assignees)
    user = raw_user or {}
    return {
        "number": raw_number,
        "title": title,
        "body": _as_str(body),
        "state": _normalize_state(state),
        "url": raw_url,
        "is_pull_request": is_pull_request,
        "user": _as_str(user.get("login") if isinstance(user, dict) else ""),
        "author_association": _as_str(raw.get("author_association")),
        "labels": labels,
        "assignees": assignees,
        "created_at": _as_str(raw.get("created_at")),
        "updated_at": _as_str(raw.get("updated_at")),
    }


def _normalize_comment(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize a raw REST comment payload to the stable output shape."""
    user = raw.get("user") or {}
    return {
        "id": int(raw.get("id", 0)),
        "user": _as_str(user.get("login") if isinstance(user, dict) else ""),
        "author_association": _as_str(raw.get("author_association")),
        "created_at": _as_str(raw.get("created_at")),
        "updated_at": _as_str(raw.get("updated_at")),
        "url": _as_str(raw.get("html_url", raw.get("url", ""))),
        "body": _as_str(raw.get("body")),
    }


def fetch_issue(number: int, *, repo: str = DEFAULT_REPO) -> dict[str, Any]:
    """Fetch a single issue via REST and return the normalized payload.

    On failure returns ``{"number": number, "status": "error", "error": ...}``
    rather than raising, mirroring :func:`scripts.dev.snapshot_issue_batch.fetch_issue`.
    """
    result = _gh_api(f"repos/{repo}/issues/{number}")
    data, error = _parse_json(result, what=f"issue {number}")
    if error:
        return {"number": number, "status": "error", "error": error}
    if not isinstance(data, dict):
        return {
            "number": number,
            "status": "error",
            "error": f"issue {number} payload was not an object",
        }
    try:
        payload = _normalize_issue(data)
        validate_issue_identity(payload, repo=repo, number=number)
    except (TypeError, ValueError) as exc:
        return {
            "number": number,
            "status": "error",
            "error": f"issue {number} payload is malformed: {exc}",
        }
    if payload["number"] != number:
        return {
            "number": number,
            "status": "error",
            "error": (
                f"issue {number} payload is malformed: "
                f"number does not match requested issue ({payload['number']})"
            ),
        }
    payload["status"] = "ok"
    return payload


def fetch_comments(
    number: int,
    *,
    repo: str = DEFAULT_REPO,
    max_pages: int = DEFAULT_MAX_COMMENT_PAGES,
) -> dict[str, Any]:
    """Fetch all comments for an issue via REST, paginating up to ``max_pages``.

    Returns ``{"status": "ok", "comments": [...]}`` on success. Fails closed with
    ``{"status": "error", "error": ...}`` when the REST read fails, the payload
    is malformed, or the comment count exceeds the page budget (so autonomous
    workflows never silently truncate a long thread).
    """
    if max_pages < 1:
        return {"status": "error", "error": f"max_pages must be >= 1, got {max_pages}"}
    comments: list[dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        # Query parameters go in the path for a GET: gh api treats --field as a
        # request body (POST/JSON), which would 422 on this read-only endpoint.
        result = _gh_api(
            f"repos/{repo}/issues/{number}/comments?per_page={COMMENTS_PAGE_SIZE}&page={page}",
        )
        data, error = _parse_json(result, what=f"comments page {page} for issue {number}")
        if error:
            return {"status": "error", "error": error}
        if not isinstance(data, list):
            return {
                "status": "error",
                "error": f"comments payload for issue {number} page {page} was not a list",
            }
        page_items = [_normalize_comment(item) for item in data if isinstance(item, dict)]
        comments.extend(page_items)
        if len(page_items) < COMMENTS_PAGE_SIZE:
            return {"status": "ok", "comments": comments}
    # Exhausted the page budget with a full last page: there may be more comments.
    return {
        "status": "error",
        "error": (
            f"issue {number} has more than {max_pages * COMMENTS_PAGE_SIZE} comments; "
            f"increase --max-comment-pages to read the full thread"
        ),
    }


def fetch_issue_with_comments(
    number: int,
    *,
    repo: str = DEFAULT_REPO,
    max_comment_pages: int = DEFAULT_MAX_COMMENT_PAGES,
) -> dict[str, Any]:
    """Fetch an issue together with all of its comments via REST.

    The returned payload has the normalized issue fields plus a ``comments`` list
    and a top-level ``status`` of ``ok``. On any failure, returns
    ``{"number": number, "status": "error", "error": ...}``.
    """
    issue = fetch_issue(number, repo=repo)
    if issue.get("status") != "ok":
        return issue
    comment_result = fetch_comments(number, repo=repo, max_pages=max_comment_pages)
    if comment_result.get("status") != "ok":
        return {
            "number": number,
            "status": "error",
            "error": str(comment_result.get("error", "unknown comments error")),
        }
    issue["comments"] = comment_result["comments"]
    return issue


def render_issue_plain(payload: dict[str, Any]) -> str:
    """Render a normalized issue-with-comments payload as a gh-like thread.

    Intended as a drop-in for ``gh issue view <number> --comments`` plain output
    in shell pipelines that expect human-readable text rather than JSON.
    """
    title = payload.get("title", "")
    state = payload.get("state", "")
    url = payload.get("url", "")
    author = payload.get("user", "")
    association = payload.get("author_association", "")
    labels = payload.get("labels", []) or []
    body = payload.get("body", "") or ""
    lines: list[str] = []
    lines.append(f"title:\t{title}")
    lines.append(f"state:\t{state}")
    if association:
        lines.append(f"association:\t{association}")
    if author:
        lines.append(f"author:\t{author}")
    if labels:
        lines.append("labels:\t" + ", ".join(labels))
    lines.append(f"url:\t{url}")
    lines.append("--")
    lines.append(body.rstrip())
    for comment in payload.get("comments", []) or []:
        c_author = comment.get("user", "")
        c_assoc = comment.get("author_association", "")
        c_created = comment.get("created_at", "")
        c_body = (comment.get("body", "") or "").rstrip()
        header = c_author
        if c_assoc:
            header = f"{c_author} ({c_assoc})"
        if c_created:
            header = f"{header} commented on {c_created}"
        lines.append("--")
        lines.append(header)
        lines.append("--")
        lines.append(c_body)
    return "\n".join(lines).rstrip() + "\n"


def _is_fallback_eligible(native_error: str) -> bool:
    """Return True when a native-first failure is safe to retry via REST.

    Fallback is only for GraphQL-path unavailability (deprecated field, quota
    exhaustion, secondary rate limit, generic GraphQL error). Authentication,
    authorization, repository-resolution, and malformed-response failures are
    deliberately NOT fallback-eligible so they fail closed instead of masking a
    real problem behind an unrelated REST read.
    """
    return is_fallback_eligible(native_error, helper=TRANSPORT_CONTRACT.helper)


def read_complete_issue_thread(
    number: int,
    *,
    repo: str = DEFAULT_REPO,
    max_comment_pages: int = DEFAULT_MAX_COMMENT_PAGES,
) -> dict[str, Any]:
    """Read a complete issue thread, falling back to REST for ``projectCards`` failures.

    Other native failures remain errors so authentication, authorization, and
    connectivity problems are not masked by an unrelated fallback path.
    """
    native = _gh_issue_view(number, repo=repo)
    if native.returncode == 0:
        return {
            "number": number,
            "status": "ok",
            "source": "gh_issue_view",
            "text": native.stdout,
        }

    native_error = (
        "\n".join(output for output in (native.stderr.strip(), native.stdout.strip()) if output)
        or f"gh issue view exited with code {native.returncode}"
    )
    if not _is_fallback_eligible(native_error):
        return {
            "number": number,
            "status": "error",
            "source": "gh_issue_view",
            "error": f"issue {number} thread read failed: {native_error}",
        }

    payload = fetch_issue_with_comments(
        number,
        repo=repo,
        max_comment_pages=max_comment_pages,
    )
    if payload.get("status") != "ok":
        fallback_error = str(payload.get("error", "unknown REST fallback error"))
        return {
            "number": number,
            "status": "error",
            "source": "rest_fallback",
            "error": (
                f"issue {number} native thread read hit GraphQL fallback-eligible error "
                f"({native_error}); REST fallback failed: {fallback_error}"
            ),
        }
    return {
        "number": number,
        "status": "ok",
        "source": "rest_fallback",
        "text": render_issue_plain(payload),
    }


def _select_fields(payload: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    """Return only the requested fields from a normalized payload."""
    if not fields:
        return payload
    known = set(ISSUE_FIELDS) | {"comments", "status", "number"}
    unknown = [field for field in fields if field not in known]
    if unknown:
        raise ValueError(f"unknown field(s): {', '.join(unknown)}")
    return {field: payload[field] for field in fields if field in payload}


def _cmd_view(args: argparse.Namespace) -> int:
    """Implement the ``view`` subcommand."""
    include_comments = args.comments or "comments" in args.fields
    if include_comments:
        payload = fetch_issue_with_comments(
            args.number,
            repo=args.repo,
            max_comment_pages=args.max_comment_pages,
        )
    else:
        payload = fetch_issue(args.number, repo=args.repo)
    if payload.get("status") != "ok":
        print(payload.get("error", "unknown error"), file=sys.stderr)
        return 1
    # Comments are opt-in (mirroring `gh issue view`, which omits them without
    # --comments). Keep them when --comments is passed, or when --json explicitly
    # requests the "comments" field (so `--json comments` returns the thread, not {}).
    if not args.comments and "comments" not in args.fields:
        payload.pop("comments", None)
    if args.plain:
        sys.stdout.write(render_issue_plain(payload))
        return 0
    try:
        selected = _select_fields(payload, args.fields)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(selected, indent=2, ensure_ascii=False))
    return 0


def _cmd_thread(args: argparse.Namespace) -> int:
    """Implement the native-first complete-thread command."""
    result = read_complete_issue_thread(
        args.number,
        repo=args.repo,
        max_comment_pages=args.max_comment_pages,
    )
    if result.get("status") != "ok":
        print(result.get("error", "unknown error"), file=sys.stderr)
        return 1
    sys.stdout.write(str(result["text"]))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="gh_issue_rest.py",
        description=(
            "GitHub issue-with-comments helper with a targeted REST fallback for "
            "the deprecated classic-Projects GraphQL field (issues #5021 and #5092)."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    thread = sub.add_parser(
        "thread",
        help="Read a complete thread via gh issue view, with targeted REST fallback.",
    )
    thread.add_argument("number", type=int, help="Issue number.")
    thread.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help=f"owner/repo to read (default: {DEFAULT_REPO}).",
    )
    thread.add_argument(
        "--max-comment-pages",
        type=int,
        default=DEFAULT_MAX_COMMENT_PAGES,
        help=f"Maximum REST fallback pages to read (each {COMMENTS_PAGE_SIZE} comments).",
    )
    thread.set_defaults(func=_cmd_thread)

    view = sub.add_parser("view", help="Read an issue and (optionally) its comments via REST.")
    view.add_argument("number", type=int, help="Issue number.")
    view.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help=f"owner/repo to read (default: {DEFAULT_REPO}).",
    )
    view.add_argument(
        "--comments",
        action="store_true",
        help="Include the comments thread (also included when --json requests 'comments').",
    )
    view.add_argument(
        "--json",
        dest="fields",
        default=[],
        nargs="*",
        metavar="FIELD",
        help="Emit only these JSON fields (space-separated). Implies JSON output.",
    )
    view.add_argument(
        "--plain",
        action="store_true",
        help="Render a gh-like human-readable thread instead of JSON.",
    )
    view.add_argument(
        "--max-comment-pages",
        type=int,
        default=DEFAULT_MAX_COMMENT_PAGES,
        help=f"Maximum comment pages to read (each {COMMENTS_PAGE_SIZE} comments).",
    )
    view.set_defaults(func=_cmd_view)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
