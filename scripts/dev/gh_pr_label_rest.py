#!/usr/bin/env python3
"""Add or remove issue/PR labels through GitHub's REST API.

Why this exists
---------------
``gh pr edit --add-label`` and ``gh issue edit --label`` can fail on GitHub CLI
versions that query the retired Projects Classic GraphQL field. This helper uses
only ``POST /repos/{owner}/{repo}/issues/{number}/labels`` and
``DELETE /repos/{owner}/{repo}/issues/{number}/labels/{label}``
and verifies that GitHub actually applied or removed the requested label. It is
deliberately REST-only: authentication, authorization, malformed responses, and
verification mismatches fail closed.

The REST issues-labels endpoint works for both issues and PRs because GitHub
treats PRs as issues for labeling. One helper covers ``gh pr edit --add-label``
and ``gh issue edit --label``.

Usage
-----
::

    uv run python scripts/dev/gh_pr_label_rest.py add 5220 \\
        --label cheap-lane --repo ll7/robot_sf_ll7

    uv run python scripts/dev/gh_pr_label_rest.py remove 5220 \\
        --label cheap-lane --repo ll7/robot_sf_ll7
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any

DEFAULT_REPO = "ll7/robot_sf_ll7"


def _gh_api_post(
    path: str, payload: dict[str, object], *, timeout: int = 30
) -> subprocess.CompletedProcess[str]:
    """POST *path* through ``gh api``, returning failures for clear handling."""
    args = ["gh", "api", "--method", "POST", path, "--input", "-"]
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
            stderr=f"gh api timed out after {timeout} seconds; label update was not verified",
        )


def _gh_api_delete(path: str, *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    """DELETE *path* through ``gh api``, returning failures for clear handling."""
    args = ["gh", "api", "--method", "DELETE", path]
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
            stderr=f"gh api timed out after {timeout} seconds; label update was not verified",
        )


def _gh_api_get(path: str, *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    """GET *path* through ``gh api`` for reading current label state."""
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
            stderr=f"gh api timed out after {timeout} seconds; could not read labels",
        )


def _get_label_names(number: int, *, repo: str = DEFAULT_REPO, timeout: int = 30) -> dict[str, Any]:
    """Return the current label names for *number* as a list, or an error dict."""
    path = f"repos/{repo}/issues/{number}/labels"
    result = _gh_api_get(path, timeout=timeout)
    if result.returncode != 0:
        detail = result.stderr.strip() or f"gh api exited with code {result.returncode}"
        return {"status": "error", "error": f"could not read labels: {detail}"}
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        snippet = result.stdout.strip()[:200]
        return {
            "status": "error",
            "error": f"label response was not valid JSON: {exc}; stdout snippet: {snippet!r}",
        }
    if not isinstance(data, list):
        return {
            "status": "error",
            "error": f"expected a list from labels endpoint, got {type(data).__name__}",
        }
    names = []
    for entry in data:
        if isinstance(entry, dict):
            name = entry.get("name")
            if isinstance(name, str):
                names.append(name)
    return {"status": "ok", "labels": names}


def add_label(number: int, label: str, *, repo: str = DEFAULT_REPO) -> dict[str, Any]:
    """Add *label* to issue/PR *number* and verify it was applied.

    Returns a compact success or error payload rather than raising so shell callers
    receive a deterministic exit status and an actionable error message.
    """
    if number < 1:
        return {"status": "error", "error": f"issue/PR number must be positive, got {number}"}
    if not label:
        return {"status": "error", "error": "label must be a non-empty string"}

    path = f"repos/{repo}/issues/{number}/labels"
    result = _gh_api_post(path, {"labels": [label]})
    if result.returncode != 0:
        detail = result.stderr.strip() or f"gh api exited with code {result.returncode}"
        return {"status": "error", "error": f"label add failed: {detail}"}
    try:
        json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        snippet = result.stdout.strip()[:200]
        return {
            "status": "error",
            "error": f"label add returned invalid JSON: {exc}; stdout snippet: {snippet!r}",
        }

    # Verify the label was actually applied by re-reading labels.
    current = _get_label_names(number, repo=repo)
    if current["status"] == "error":
        return current
    if label not in current["labels"]:
        return {
            "status": "error",
            "error": f"label '{label}' was not found in labels after add; "
            "the write may not have taken effect",
        }
    return {
        "status": "ok",
        "number": number,
        "label": label,
        "action": "add",
        "repo": repo,
    }


def remove_label(number: int, label: str, *, repo: str = DEFAULT_REPO) -> dict[str, Any]:
    """Remove *label* from issue/PR *number* and verify it was removed.

    Returns a compact success or error payload rather than raising so shell callers
    receive a deterministic exit status and an actionable error message.
    """
    if number < 1:
        return {"status": "error", "error": f"issue/PR number must be positive, got {number}"}
    if not label:
        return {"status": "error", "error": "label must be a non-empty string"}

    path = f"repos/{repo}/issues/{number}/labels/{label}"
    result = _gh_api_delete(path)
    if result.returncode != 0:
        detail = result.stderr.strip() or f"gh api exited with code {result.returncode}"
        return {"status": "error", "error": f"label remove failed: {detail}"}

    # Verify the label was actually removed by re-reading labels.
    current = _get_label_names(number, repo=repo)
    if current["status"] == "error":
        return current
    if label in current["labels"]:
        return {
            "status": "error",
            "error": f"label '{label}' was still found in labels after remove; "
            "the delete may not have taken effect",
        }
    return {
        "status": "ok",
        "number": number,
        "label": label,
        "action": "remove",
        "repo": repo,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("add", "remove"),
        help="Whether to add or remove the label.",
    )
    parser.add_argument("number", type=int, help="Issue or PR number to update.")
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help=f"owner/repo to update (default: {DEFAULT_REPO}).",
    )
    parser.add_argument("--label", required=True, help="Label name to add or remove.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the label helper and emit one compact JSON result."""
    args = _build_parser().parse_args(argv)
    if args.action == "add":
        result = add_label(args.number, args.label, repo=args.repo)
    else:
        result = remove_label(args.number, args.label, repo=args.repo)

    stream = sys.stdout if result["status"] == "ok" else sys.stderr
    print(json.dumps(result, sort_keys=True), file=stream)
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
