#!/usr/bin/env python3
"""Verify that a PR's closing issue references include an expected issue number.

Queries GitHub GraphQL via ``gh api graphql`` to retrieve
``repository.pullRequest.closingIssuesReferences`` and fails closed unless the
expected issue is among them.

Exit codes:
    0  Expected issue is among the PR's closing references.
    1  Expected issue is NOT among the PR's closing references.
    2  Could not determine (API error, unavailable, or missing input).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

CLOSING_ISSUES_QUERY = """
query($owner: String!, $repo: String!, $prNumber: Int!) {
  repository(owner: $owner, name: $repo) {
    pullRequest(number: $prNumber) {
      closingIssuesReferences(first: 100) {
        nodes {
          number
          url
        }
        pageInfo {
          hasNextPage
        }
      }
    }
  }
}
"""


@dataclass(frozen=True)
class ClosingReferenceResult:
    """Compact result for a PR closing-reference verification."""

    status: str
    pr_number: str
    expected_issue: int
    actual_closing_issues: tuple[int, ...]
    message: str


def _gh(args: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a gh command and return the completed process."""
    return subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _parse_json(stdout: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse gh JSON stdout into a dict or an error string."""
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as exc:
        return None, f"Failed to parse gh output as JSON: {exc}"
    if not isinstance(data, dict):
        return None, "gh output is not a JSON object"
    return data, None


def _resolve_repo(explicit: str) -> str | None:
    """Resolve the repository identifier, auto-detecting from gh when empty."""
    if explicit:
        return explicit
    result = _gh(["repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner"])
    if result.returncode != 0:
        return None
    repo = result.stdout.strip()
    return repo if repo else None


def _split_owner_repo(repo: str) -> tuple[str, str]:
    """Split owner/repo into (owner, repo)."""
    parts = repo.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Invalid repository format: {repo!r}; expected owner/repo")
    return parts[0], parts[1]


def _extract_graphql_error(payload: dict[str, Any]) -> str | None:
    """Return a normalized top-level GraphQL error, if present."""
    errors = payload.get("errors")
    if errors is None:
        return None
    if not isinstance(errors, list):
        return "GraphQL errors field is malformed"
    if not errors:
        return None
    messages = [e.get("message", str(e)) for e in errors if isinstance(e, dict)]
    return "; ".join(messages) if messages else "GraphQL returned errors"


def _extract_closing_nodes(closing_refs: dict[str, Any]) -> tuple[list[int] | None, str | None]:
    """Validate and extract one complete closing-reference connection."""
    nodes = closing_refs.get("nodes")
    if not isinstance(nodes, list):
        return None, "closingIssuesReferences.nodes is not a list"
    issues: list[int] = []
    for index, node in enumerate(nodes):
        if not isinstance(node, dict) or type(node.get("number")) is not int:
            return None, f"closingIssuesReferences.nodes[{index}] is malformed"
        issues.append(node["number"])
    page_info = closing_refs.get("pageInfo")
    if not isinstance(page_info, dict) or not isinstance(page_info.get("hasNextPage"), bool):
        return None, "closingIssuesReferences.pageInfo is malformed"
    if page_info["hasNextPage"]:
        return None, "More than 100 closing references exist; refusing an incomplete result"
    return issues, None


def _extract_issues_from_graphql(
    payload: dict[str, Any],
    pr_number: str,
) -> tuple[list[int] | None, str | None]:
    """Extract closing issue numbers from a parsed GraphQL response."""
    graphql_error = _extract_graphql_error(payload)
    if graphql_error is not None:
        return None, graphql_error
    data = payload.get("data")
    if not isinstance(data, dict):
        return None, "GraphQL response is missing data field"
    repo_data = data.get("repository")
    if not isinstance(repo_data, dict):
        return None, "Repository not found in GraphQL response"
    pr_data = repo_data.get("pullRequest")
    if not isinstance(pr_data, dict):
        return None, f"PR #{pr_number} not found"
    closing_refs = pr_data.get("closingIssuesReferences")
    if not isinstance(closing_refs, dict):
        return None, "closingIssuesReferences is not available"
    return _extract_closing_nodes(closing_refs)


def _fetch_closing_issues(
    pr_number: str,
    *,
    repo: str,
) -> tuple[list[int] | None, str | None]:
    """Return the list of closing issue numbers for a PR, or (None, error)."""
    try:
        owner, repo_name = _split_owner_repo(repo)
    except ValueError as exc:
        return None, str(exc)
    try:
        pr_int = int(pr_number)
    except ValueError:
        return None, f"Invalid PR number: {pr_number!r}; expected a positive integer"
    if pr_int < 1:
        return None, f"Invalid PR number: {pr_number!r}; expected a positive integer"
    result = _gh(
        [
            "api",
            "graphql",
            "-f",
            f"query={CLOSING_ISSUES_QUERY}",
            "-F",
            f"owner={owner}",
            "-F",
            f"repo={repo_name}",
            "-F",
            f"prNumber={pr_int}",
        ],
        timeout=30,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        return None, f"gh api graphql failed: {stderr or result.stdout.strip()}"
    payload, parse_error = _parse_json(result.stdout)
    if parse_error or payload is None:
        return None, parse_error or "Failed to parse GraphQL response"
    return _extract_issues_from_graphql(payload, pr_number)


def check_closing_reference(
    pr_number: str, expected_issue: int, *, repo: str
) -> ClosingReferenceResult:
    """Check whether a PR's closing references include the expected issue."""
    if expected_issue < 1:
        return ClosingReferenceResult(
            status="error",
            pr_number=pr_number,
            expected_issue=expected_issue,
            actual_closing_issues=(),
            message=f"Invalid expected issue: {expected_issue!r}; expected a positive integer",
        )
    issues, error = _fetch_closing_issues(pr_number, repo=repo)
    if error is not None or issues is None:
        return ClosingReferenceResult(
            status="error",
            pr_number=pr_number,
            expected_issue=expected_issue,
            actual_closing_issues=(),
            message=error or "Unknown error fetching closing issues",
        )
    actual = tuple(sorted(issues))
    if expected_issue in actual:
        return ClosingReferenceResult(
            status="ok",
            pr_number=pr_number,
            expected_issue=expected_issue,
            actual_closing_issues=actual,
            message=f"PR #{pr_number} closes #{expected_issue}",
        )
    return ClosingReferenceResult(
        status="mismatch",
        pr_number=pr_number,
        expected_issue=expected_issue,
        actual_closing_issues=actual,
        message=(
            f"PR #{pr_number} does not close #{expected_issue}. "
            f"Actual closing references: {actual or '(none)'}."
        ),
    )


def _format_human(result: ClosingReferenceResult) -> str:
    actual = ", ".join(f"#{n}" for n in result.actual_closing_issues) or "(none)"
    return (
        f"PR #{result.pr_number} closing-reference check: "
        f"status={result.status}; "
        f"expected=#{result.expected_issue}; "
        f"actual=[{actual}]; "
        f"{result.message}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pr_number", help="GitHub PR number to check")
    parser.add_argument("expected_issue", type=int, help="Expected closing issue number")
    parser.add_argument("--repo", default="", help="owner/repo (default: detect from gh)")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    return parser


def _emit_json(
    status: str, pr_number: str, expected_issue: int, actual: list[int], message: str
) -> None:
    """Print a JSON result object to stdout."""
    print(
        json.dumps(
            {
                "status": status,
                "pr_number": pr_number,
                "expected_issue": expected_issue,
                "actual_closing_issues": actual,
                "message": message,
            }
        )
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point: check PR closing reference and print results."""
    args = _build_parser().parse_args(argv)
    try:
        repo = _resolve_repo(args.repo)
        result = (
            check_closing_reference(args.pr_number, args.expected_issue, repo=repo)
            if repo is not None
            else None
        )
    except subprocess.TimeoutExpired:
        msg = "gh CLI command timed out."
        if args.json:
            _emit_json("error", args.pr_number, args.expected_issue, [], msg)
        else:
            print(msg, file=sys.stderr)
        return 2
    except OSError as exc:
        msg = f"Failed to execute gh CLI: {exc}"
        if args.json:
            _emit_json("error", args.pr_number, args.expected_issue, [], msg)
        else:
            print(msg, file=sys.stderr)
        return 2
    if repo is None:
        msg = "Failed to detect repository.  Pass --repo owner/repo."
        if args.json:
            _emit_json("error", args.pr_number, args.expected_issue, [], msg)
        else:
            print(msg, file=sys.stderr)
        return 2
    assert result is not None
    if args.json:
        _emit_json(
            result.status,
            result.pr_number,
            result.expected_issue,
            list(result.actual_closing_issues),
            result.message,
        )
    else:
        print(_format_human(result))
    if result.status == "error":
        return 2
    if result.status == "mismatch":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
