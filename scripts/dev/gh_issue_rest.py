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
known ``repository.issue.projectCards`` failure. The existing ``view`` command
and library functions remain explicit REST-backed interfaces.

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

DEFAULT_REPO = "ll7/robot_sf_ll7"
DEFAULT_MAX_COMMENT_PAGES = 10
COMMENTS_PAGE_SIZE = 100
PROJECT_CARDS_ERROR_MARKER = "repository.issue.projectCards"

# Fields exposed in the normalized issue payload, in a stable order.
ISSUE_FIELDS = (
    "number",
    "title",
    "body",
    "state",
    "url",
    "user",
    "author_association",
    "labels",
    "assignees",
    "created_at",
    "updated_at",
)
COMMENT_FIELDS = ("id", "user", "author_association", "created_at", "updated_at", "url", "body")


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


def _gh_api(
    path: str, *, params: list[str] | None = None, timeout: int = 30
) -> subprocess.CompletedProcess:
    """Run a ``gh api`` REST read and return the completed process.

    Failures are returned (not raised) so callers can render clear errors. The
    params list lets callers add ``--field``/``-q`` style ``gh api`` flags.
    """
    args = ["gh", "api", path]
    if params:
        args.extend(params)
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError:
        # gh CLI not installed / not on PATH. Return a failed result instead of
        # raising so callers keep the documented "returns error payload" contract.
        return subprocess.CompletedProcess(
            args=args,
            returncode=127,
            stdout="",
            stderr="gh CLI not found on PATH; install GitHub CLI (https://cli.github.com/)",
        )


def _parse_json(
    result: subprocess.CompletedProcess, *, what: str
) -> tuple[dict[str, Any] | list[Any] | None, str]:
    """Parse JSON from a ``gh api`` result, returning ``(data, error)``.

    On failure ``data`` is ``None`` and ``error`` is a human-readable message.
    """
    if result.returncode != 0:
        detail = result.stderr.strip() or f"gh api exited with code {result.returncode}"
        return None, f"{what} failed: {detail}"
    try:
        return json.loads(result.stdout), ""
    except json.JSONDecodeError as exc:
        snippet = result.stdout.strip()[:200]
        return None, f"{what} returned invalid JSON: {exc}; stdout snippet: {snippet!r}"


def _as_str(raw: Any) -> str:
    """Coerce a JSON value to ``str``, mapping explicit ``None`` to ``""``.

    Guards against ``str(None) -> "None"`` when a REST field is present but
    ``null`` (e.g. an issue with an empty body), while preserving valid falsy
    values such as ``0`` or ``""``.
    """
    return "" if raw is None else str(raw)


def _normalize_state(raw: Any) -> str:
    """Return an uppercase state string matching ``gh issue view --json``."""
    return str(raw).upper() if raw else ""


def _normalize_issue(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize a raw REST issue payload to the stable output shape."""
    labels = sorted(
        str(label.get("name", ""))
        for label in raw.get("labels", [])
        if isinstance(label, dict) and label.get("name")
    )
    assignees = sorted(
        str(user.get("login", ""))
        for user in raw.get("assignees", [])
        if isinstance(user, dict) and user.get("login")
    )
    user = raw.get("user") or {}
    return {
        "number": int(raw.get("number", 0)),
        "title": _as_str(raw.get("title")),
        "body": _as_str(raw.get("body")),
        "state": _normalize_state(raw.get("state", "")),
        "url": _as_str(raw.get("html_url", raw.get("url", ""))),
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
    payload = _normalize_issue(data)
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
    if PROJECT_CARDS_ERROR_MARKER not in native_error:
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
                f"issue {number} native thread read hit {PROJECT_CARDS_ERROR_MARKER}; "
                f"REST fallback failed: {fallback_error}"
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
    payload = fetch_issue_with_comments(
        args.number,
        repo=args.repo,
        max_comment_pages=args.max_comment_pages,
    )
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
