#!/usr/bin/env python3
"""Update or reconcile a pull-request title/body through GitHub's REST API.

Why this exists
---------------
``gh pr edit --body-file`` can fail on GitHub CLI versions that query the retired
Projects Classic GraphQL field. This helper uses only ``PATCH /repos/{owner}/{repo}/pulls/{number}``
and verifies that GitHub returned the requested body. It is deliberately REST-only:
authentication, authorization, malformed responses, and body mismatches fail closed.
The ``--reconcile`` mode reads the current title/body first, performs one atomic
title-and-body PATCH only when either field differs, and verifies both fields.

Usage
-----
::

    uv run python scripts/dev/gh_pr_body_rest.py 5220 \
        --repo ll7/robot_sf_ll7 --body-file /tmp/pr-body.md

    uv run python scripts/dev/gh_pr_body_rest.py 5220 --reconcile \
        --title "fix: final title" --repo ll7/robot_sf_ll7 \
        --body-file /tmp/pr-body.md
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from scripts.dev.pr_metadata import metadata_digest, validate_pr_title

DEFAULT_REPO = "ll7/robot_sf_ll7"


def _gh_api_patch(
    path: str, payload: dict[str, str], *, timeout: int = 30
) -> subprocess.CompletedProcess[str]:
    """Patch *path* through ``gh api``, returning failures for clear handling."""
    args = ["gh", "api", "--method", "PATCH", path, "--input", "-"]
    try:
        return subprocess.run(
            args,
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
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
            stderr=f"gh api timed out after {timeout} seconds; body update was not verified",
        )


def _gh_api_get(path: str, *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    """Fetch *path* through ``gh api``, returning failures for clear handling."""
    args = ["gh", "api", path]
    try:
        return subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
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
            stderr=f"gh api timed out after {timeout} seconds; PR metadata was not verified",
        )


def _read_body_file(body_file: Path) -> tuple[str | None, str | None]:
    """Read a UTF-8 PR body file, returning ``(body, error)``."""
    try:
        return body_file.read_text(encoding="utf-8"), None
    except OSError as exc:
        return None, f"could not read body file {body_file}: {exc}"


def _decode_object(
    result: subprocess.CompletedProcess[str], *, operation: str
) -> tuple[dict[str, Any] | None, str | None]:
    """Decode a successful ``gh api`` result as a JSON object."""
    if result.returncode != 0:
        detail = result.stderr.strip() or f"gh api exited with code {result.returncode}"
        return None, f"{operation} failed: {detail}"
    try:
        response = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        snippet = result.stdout.strip()[:200]
        return None, f"{operation} returned invalid JSON: {exc}; stdout snippet: {snippet!r}"
    if not isinstance(response, dict):
        return None, f"{operation} response was not an object"
    return response, None


def update_pr_body(number: int, body_file: Path, *, repo: str = DEFAULT_REPO) -> dict[str, Any]:
    """Update PR *number* from *body_file* and verify the REST response.

    Returns a compact success or error payload rather than raising so shell callers
    receive a deterministic exit status and an actionable error message.
    """
    if number < 1:
        return {"status": "error", "error": f"PR number must be positive, got {number}"}
    body, body_error = _read_body_file(body_file)
    if body_error:
        return {"status": "error", "error": body_error}
    assert body is not None

    result = _gh_api_patch(f"repos/{repo}/pulls/{number}", {"body": body})
    if result.returncode != 0:
        detail = result.stderr.strip() or f"gh api exited with code {result.returncode}"
        return {"status": "error", "error": f"PR body update failed: {detail}"}
    try:
        response = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        snippet = result.stdout.strip()[:200]
        return {
            "status": "error",
            "error": f"PR body update returned invalid JSON: {exc}; stdout snippet: {snippet!r}",
        }
    if not isinstance(response, dict):
        return {"status": "error", "error": "PR body update response was not an object"}
    if response.get("body") != body:
        return {
            "status": "error",
            "error": "PR body update response did not preserve the requested body",
        }
    return {
        "status": "ok",
        "number": number,
        "repo": repo,
        "url": str(response.get("html_url", "")),
    }


def reconcile_pr_metadata(  # noqa: C901
    number: int,
    title: str,
    body_file: Path,
    *,
    repo: str = DEFAULT_REPO,
) -> dict[str, Any]:
    """Reconcile a PR's final title/body with one verified atomic REST update.

    The current PR is read before mutation. A matching title/body returns an
    explicit no-op; otherwise one PATCH carries both fields and the response
    must preserve both exact requested strings.
    """
    if number < 1:
        return {"status": "error", "error": f"PR number must be positive, got {number}"}
    title_error = validate_pr_title(title)
    if title_error:
        return {"status": "error", "error": title_error}
    body, body_error = _read_body_file(body_file)
    if body_error:
        return {"status": "error", "error": body_error}
    assert body is not None

    current_result = _gh_api_get(f"repos/{repo}/pulls/{number}")
    current, current_error = _decode_object(current_result, operation="PR metadata read")
    if current_error:
        return {"status": "error", "error": current_error}
    assert current is not None
    current_title = current.get("title")
    current_body = current.get("body")
    if not isinstance(current_title, str):
        return {"status": "error", "error": "PR metadata read returned a malformed title"}
    if current_body is None:
        current_body = ""
    if not isinstance(current_body, str):
        return {"status": "error", "error": "PR metadata read returned a malformed body"}

    desired_digest = metadata_digest(title, body)
    current_digest = metadata_digest(current_title, current_body)
    changed_fields = [
        field
        for field, current_value, desired_value in (
            ("title", current_title, title),
            ("body", current_body, body),
        )
        if current_value != desired_value
    ]
    base_result = {
        "number": number,
        "repo": repo,
        "url": str(current.get("html_url", "")),
        "metadata_digest": desired_digest,
        "previous_metadata_digest": current_digest,
        "changed_fields": changed_fields,
    }
    if not changed_fields:
        return {"status": "unchanged", **base_result, "changed": False}

    patch_result = _gh_api_patch(
        f"repos/{repo}/pulls/{number}",
        {"title": title, "body": body},
    )
    response, patch_error = _decode_object(patch_result, operation="PR metadata update")
    if patch_error:
        return {"status": "error", "error": patch_error}
    assert response is not None
    if response.get("title") != title or response.get("body") != body:
        return {
            "status": "error",
            "error": "PR metadata update response did not preserve the requested title and body",
        }
    return {
        "status": "ok",
        **base_result,
        "url": str(response.get("html_url", current.get("html_url", ""))),
        "changed": True,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("number", type=int, help="Pull-request number to update.")
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help=f"owner/repo to update (default: {DEFAULT_REPO}).",
    )
    parser.add_argument(
        "--body-file", type=Path, required=True, help="Markdown body file to apply."
    )
    parser.add_argument(
        "--reconcile",
        action="store_true",
        help="Read current metadata and atomically reconcile title and body when needed.",
    )
    parser.add_argument(
        "--title",
        help="Final PR title; required with --reconcile.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the PR-body updater and emit one compact JSON result."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.reconcile and args.title is None:
        parser.error("--reconcile requires --title")
    if not args.reconcile and args.title is not None:
        parser.error("--title requires --reconcile")
    if args.reconcile:
        result = reconcile_pr_metadata(
            args.number,
            args.title,
            args.body_file,
            repo=args.repo,
        )
    else:
        result = update_pr_body(args.number, args.body_file, repo=args.repo)
    success = result["status"] in {"ok", "unchanged"}
    stream = sys.stdout if success else sys.stderr
    print(json.dumps(result, sort_keys=True), file=stream)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
