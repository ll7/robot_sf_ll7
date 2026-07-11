#!/usr/bin/env python3
"""Update a pull-request body through GitHub's REST API.

Why this exists
---------------
``gh pr edit --body-file`` can fail on GitHub CLI versions that query the retired
Projects Classic GraphQL field. This helper uses only ``PATCH /repos/{owner}/{repo}/pulls/{number}``
and verifies that GitHub returned the requested body. It is deliberately REST-only:
authentication, authorization, malformed responses, and body mismatches fail closed.

Usage
-----
::

    uv run python scripts/dev/gh_pr_body_rest.py 5220 \
        --repo ll7/robot_sf_ll7 --body-file /tmp/pr-body.md
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

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


def update_pr_body(number: int, body_file: Path, *, repo: str = DEFAULT_REPO) -> dict[str, Any]:
    """Update PR *number* from *body_file* and verify the REST response.

    Returns a compact success or error payload rather than raising so shell callers
    receive a deterministic exit status and an actionable error message.
    """
    if number < 1:
        return {"status": "error", "error": f"PR number must be positive, got {number}"}
    try:
        body = body_file.read_text(encoding="utf-8")
    except OSError as exc:
        return {"status": "error", "error": f"could not read body file {body_file}: {exc}"}

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
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the PR-body updater and emit one compact JSON result."""
    args = _build_parser().parse_args(argv)
    result = update_pr_body(args.number, args.body_file, repo=args.repo)
    stream = sys.stdout if result["status"] == "ok" else sys.stderr
    print(json.dumps(result, sort_keys=True), file=stream)
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
